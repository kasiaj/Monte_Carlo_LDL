#!/usr/bin/python 
import numpy as np 
from scipy.sparse import dia_matrix
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

import matplotlib
import matplotlib.pyplot as plt
import pycuda.gpuarray as gpuarray
from pycuda.curandom import rand as curand
from pycuda.compiler import SourceModule
import pycuda.driver as cuda

# <markdowncell>

# Parametry:

# <codecell>

nazwy       = [  'container','endothel' , 'intima']

D           =[   6e-11     ,  6e-11  , 5.4e-6 , 3.18e-9  ,  5e-8] 
V           = [   2.6783499775030101e-5    ,  2.6783499775030101e-5    , 2.6783499775030101e-5  , 2.6783499775030101e-5, 2.6783499775030101e-5 ]
sigma       = [  0.93108378484950194,     0.93108378484950194, 0.8272, 0.9827, 0.8836]
L           = [    0.2      ,  1.8        ,10. ,2.0, 200.]
k_react     = [    0.       ,  0.        , 0. , 0., 1.4e-4]

#Change units to um
D = [D_*1e6 for D_ in D]
V = [V_*1e3 for V_ in V]
dt=0.01#1.0#

# <codecell>


# <markdowncell>

# Container jest sztucznym tworem, ktory ma zapewnic stale stezenie na granicy endothelium.

# <markdowncell>

# Inicjalizacja CUDY

# <codecell>

cuda.init()
device = cuda.Device(1)
ctx = device.make_context()
print device.name(), device.compute_capability(),device.total_memory()/1024.**3,"GB"
print "a tak wogole to mamy tu:",cuda.Device.count(), " urzadzenia"

# <markdowncell>

# Definiujemy kernel:  

# <codecell>


def get_kernel(pars):

    rng_src = """
#define PI 3.14159265358979f

extern int printf(__const char *__restrict __format, ...);
/*
 * Return a uniformly distributed random number from the
 * [0;1] range.
 */
__device__ float rng_uni(unsigned int *state)
{
	unsigned int x = *state;

	x = x ^ (x >> 13);
	x = x ^ (x << 17);
	x = x ^ (x >> 5);

	*state = x;

	return x / 4294967296.0f;
}
/*
 * Generate two normal variates given two uniform variates. Przeksztalcenie Boxa-Mullera:
 */
__device__ void bm_trans(float& u1, float& u2)
{
	float r = sqrtf(-2.0f * logf(u1));
	float phi = 2.0f * PI * u2;
	u1 = r * cosf(phi);
	u2 = r * sinf(phi);
}


"""
    
    src = """

__device__  float D(float x)
{{
    if(x<{L[0]}f)
    {{
        return {D[0]}f;
    }}
    else if (x<{L[0]}f+{L[1]}f){{
        return {D[1]}f;
    }}
    else if (x<{L[0]}f+{L[1]}f+{L[2]}f){{
        return {D[2]}f;
    }}
    else if (x<{L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f){{
        return {D[3]}f;
    }}
    else
    {{
        return {D[4]}f;
    }}
}}


__device__  float sigma(float x)
{{
    if(x<{L[0]}f)
    {{
        return {s[0]}f;
    }}
    else if (x<{L[0]}f+{L[1]}f){{
        return {s[1]}f;
    }}
    else if (x<{L[0]}f+{L[1]}f+{L[2]}f){{
        return {s[2]}f;
    }}
    else if (x<{L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f){{
        return {s[3]}f;
    }}
    else
    {{
        return {s[4]}f;
    }}
}}
  
__device__  float V(float x)
{{
    if(x<{L[0]}f)
    {{
        return {V[0]}f;
    }}
    else if (x<{L[0]}f+{L[1]}f){{
        return {V[1]}f;
    }}
    else if (x<{L[0]}f+{L[1]}f+{L[2]}f){{
        return {V[2]}f;
    }}
    else if (x<{L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f){{
        return {V[3]}f;
    }}
    else{{
        return {V[4]}f;
    }}
}}
  
__device__  float k_react(float x)
{{
    if (x>{L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f)
    {{
      return {k_react[4]}f;
    }}
    else
    {{
      return -1.0;
    }}
}}
  
    
    
__global__ void advance_tex(float *x_i, unsigned int *rng_state,int sps)    
{{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    float x, x1;                                     //x1 to delta Y z publikacji
    int k;       
    float n1, n2,c1; 	                             //do losowania
    unsigned int lrng_state;                         //liczby losowe
     
    for (k=0;k<sps;k++){{
        x = x_i[idx];
        lrng_state = rng_state[idx];
        if(x>-1.0)
        {{
            //Reakcja lub pochlanianie
            c1 = rng_uni(&lrng_state);
            if(c1<=k_react(x)*{dt}f|| x>=({L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f+{L[4]}f))
            {{
                x = -1.0;
            }}
            if(x>-1.0)
            {{
                //Dyfuzja
                n1 = rng_uni(&lrng_state);
		        n2 = rng_uni(&lrng_state);
		        bm_trans(n1, n2);         //rozklad normalny
                
                x1 =fabsf( x + sqrtf({dt}f * D(x)  * 2.0f) * n1);
                if (x1>=({L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f+{L[4]}f))                    //dla x1 rowniez nalezy zastosowac warunek brzegowy
                {{
                    x=-1.0;
                }}
                else
                {{
                    x = fabsf(x + sqrtf({dt}f  *D(x1) * 2.0f) * n1 + {dt}f*(1-sigma(x))*V(x));
                }}
                if (x<={L[0]}f||x>({L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f+{L[4]}f))
                {{
                    x=-1.0;                                                           //(1)  oproznienie kontenera i warunek brzegowy dla x 
                                                                                      //gdy tracer po dyfuzji nadal znajduje sie w kontenerze lub sietam pojawil, zostaje deaktywowany, aby zapewnic pusty obszar dla generacji tracerow 
                                                                                      //wowczas nie ma koniecznosci liczenia ile tracerow znajduje sie w zadanym obszarze i ile nalezaloby uzupelnic, z gory znana jest liczba trecerow do aktywacji     
                }}
            }}
            x_i[idx] = x;
        }}
        
        __syncthreads();   
    
        if(idx%{tracers}=={tracers}-1)
        {{
            bool found = false;
            int rnd;
      
             //Poszukiwnie wolnego watku, do symulacji tracera w obszarze stalego stezenia (2)
             int tmp;
             tmp=k;
             rnd=idx+fmodf(k,{tracers})-{tracers}+1;                  //Jako pierwszy sprawdzany jest k-ty (k-numer kroku) element w danym bloku CUDY  [blok 1: 0-tracers-1, blok 2:tracers-2*tracers-1, ...]
             while(!found)
             {{
                 if (x_i[rnd]==-1.0)
                 {{
                     //znaleziona
                     found = true;
                     x_i[rnd] = rng_uni(&lrng_state)*{L[0]}f;
                 }}
                 else
                 {{   
                     tmp++;
                     rnd=idx+fmodf(tmp,{tracers})-{tracers}+1;      //Gy pierwsza proba sie nie powiodla sprawdzany jest saiedni element, z uwzglednieniem zawiniecia, aby nie opuscic biezacego bloku
                          
                 }}
             }}
        }}
             
        __syncthreads(); 
        rng_state[idx] = lrng_state;
    }}
}}
""".format(**pars)
    mod = SourceModule(rng_src+src,options=["--use_fast_math"]  )
    #print src
    advance_tex = mod.get_function("advance_tex")
    return mod,advance_tex

# <codecell>



# <codecell>


# <headingcell level=3>

# Inicjalizacja kerneli i przygotowanie symulacji

# <codecell>

Nsteps = 1024*40
blocks = 2622      #ilosc blokow odpowiada ilosci tracerow w obszarze o stalym stezeniu, a kazdym bloku symulowany jest zawsze 1 tracer z tego obszaru.(2)
                   #w kazdym watku jest zagwarantowane (1) ze po kazdym kroku symulacji kontener jest oprozniony
block_size = 300   #dobrany aby calkowita ilosc tracerow w ciagu calej symulacji byla odpowiednia


n_tracers = blocks*block_size   #polowa to martwe dusze

xinit = np.ones(n_tracers,dtype=np.float32)
xinit = xinit * (-1)
#polozenie -1 znaczy, ze dany watek nie jest jeszcze wykorzystywany w symulacji, ale jest gotowy do wykorzystania, dzieki temu symulacja nie musi byc sterowana w nadzednym watku, aby kazdorazowo zmieniac ilosc tracerow i odpowiadajacych im watkow
distance=block_size

for i in range(0, n_tracers, distance):
    xinit[i] = (L[0]) * np.random.random_sample()


tracers= ((0 < xinit) & (xinit <= L[0])).sum()
print "  tracers" , tracers

pars = {'dt':dt, 'L':L, 'D':D, 's':sigma, 'k_react':k_react, 'V':V, 'tracers':distance}
mod,advance = get_kernel(pars)



dx_i = gpuarray.to_gpu (xinit)
rng_state = np.array(np.random.randint(1,2147483648,size=n_tracers),dtype=np.uint32)
rng_state_g = gpuarray.to_gpu(rng_state)

print n_tracers,Nsteps,n_tracers*Nsteps/1e9,"G","dt=",dt

# <markdowncell>

# Uruchomienie symulacji
# -----------------------

# <codecell>

import time
test = np.int32(0)
sps = np.int32(10)#2000000 950000
size = np.int32(n_tracers)
                                   
start = time.time()
for k in range(1):
    advance(dx_i,rng_state_g, sps, block=(block_size,1,1), grid=(blocks,1))
    print ((0 < dx_i.get()) & (dx_i.get() <= L[0])).sum()
    if k%1==0:
        with open('test3.log', 'w') as f:
            f.writelines(str(j)  + '\n' for j in dx_i.get())


ctx.synchronize()
elapsed = (time.time() - start)  
print elapsed,np.nonzero(np.isnan(dx_i.get()))[0].shape[0]


ctx.pop()
ctx.detach()
