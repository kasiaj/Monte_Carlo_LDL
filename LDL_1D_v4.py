#!/usr/bin/python 
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Model akomodacji LDL'a przy wykorzystaniu metody stochastycznej.

# <markdowncell>

# WstÄ™p
# -----
# 
# 
# W tym dokumencie opisany jest sposĂłb modelowania transportu LDL w Ĺ›cianie naczynia. Program rozwiÄ…zuje problem transportu zgodnie z rĂłwnaniem:
#   
# \begin{eqnarray}
#       \frac{\partial c}{\partial t} +(1-\sigma)\vec u\cdot\nabla c & = & D_{eff} \nabla^2 c - k c
# \end{eqnarray}
# 
# PrzyjmujÄ…c jednowymiarowy model, w kierunku radialnym, prÄ™dkoĹ›Ä‡  filtracji $\vec u$ ta musi byÄ‡ staĹ‚a na caĹ‚ym odcinku filtracji przez Ĺ›cianÄ™. 
# PoniĹĽsze symulacje przyjmujÄ… wartoĹ›Ä‡ $v=2.3\cdot 10^{-5} \frac{mm}{s}=0.023 \frac{\mu m}{s}.$ Jednak w  efektywna prÄ™dkoĹ›Ä‡ 
# unoszenia jest dana wzrorem $(1-\sigma)\vec u$, gdzie $\sigma$ to wspĂłĹ‚czynnik odbicia. Oznacza to, ĹĽe w zaleĹĽnoĹ›ci od efektywnych wĹ‚asnoĹ›ci materiaĹ‚u
# w ktĂłrym poruszajÄ… siÄ™ czÄ…steczki LDL mamy rĂłĹĽne prÄ™dkoĹ›ci dryftu. 
# 
# Symulacja wykorzystuje metodÄ™ stochastycznÄ….

# <codecell>

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
#dobre
D           =[   6e-11     ,  6e-11  , 5.4e-6 , 3.18e-9  ,  5e-8] 
V           = [   2.6783499775030101e-5    ,  2.6783499775030101e-5    , 2.6783499775030101e-5  , 2.6783499775030101e-5, 2.6783499775030101e-5 ]
sigma       = [  0.93108378484950194,     0.93108378484950194, 0.8272, 0.9827, 0.8836]
L           = [    0.2      ,  1.8        ,10. ,2.0, 200.]
k_react     = [    0.       ,  0.        , 0. , 0., 1.4e-4]



#Change units to um
D = [D_*1e6 for D_ in D]
V = [V_*1e3 for V_ in V]
dt=0.01#1.0#
SD=[np.sqrt(D_*dt*2.0) for D_ in D]
VV=[dt*(1-sigma_)*V_ for sigma_, V_ in zip(sigma, V)]

# <codecell>

print VV
print V
print SD

# <markdowncell>

# Container jest sztucznym tworem, ktĂłry ma zapewniÄ‡ staĹ‚e stÄ™ĹĽenie na granicy endothelium.

# <markdowncell>

# Inicjalizacja CUDY

# <codecell>

cuda.init()
device = cuda.Device(2)
ctx = device.make_context()
print device.name(), device.compute_capability(),device.total_memory()/1024.**3,"GB"
print "a tak wogĂłle to mamy tu:",cuda.Device.count(), " urzÄ…dzenia"

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
__device__  float k_react(float x)
{{
//float x=*xx;
    if (x>{L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f)
    {{
      return {k_react[4]}f;
    }}
    else
    {{
      return -1.0;
    }}
}}
 

  
    __device__  float SD(float x)
{{
//float x=*xx;
    if(x>{L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f)
    {{
        return {SD[4]}f;
    }}
    else if (x>{L[0]}f+{L[1]}f+{L[2]}f){{
        return {SD[3]}f;
    }}
    else if (x>{L[0]}f+{L[1]}f){{
        return {SD[2]}f;
    }}
    else if (x>{L[0]}f){{
        return {SD[1]}f;
    }}
    else
    {{
        return {SD[0]}f;
    }}
}}

__device__  float V(float x)//velsigma
{{
//float x=*xx;
    if(x>{L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f)
    {{
        return {V[4]}f;
    }}
    else if (x>{L[0]}f+{L[1]}f+{L[2]}f){{
        return {V[3]}f;
    }}
    else if (x>{L[0]}f+{L[1]}f){{
        return {V[2]}f;
    }}
    else if (x>{L[0]}f){{
        return {V[1]}f;
    }}
    else
    {{
        return {V[0]}f;
    }}
}}

    
__global__ void advance_tex(float *x_i, int *indexes,unsigned int *rng_state,int sps)    
{{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    float x, x1;                                     //x1 to delta Y z publikacji
    int k;       
    float n1, n2; 	                             //do losowania
    unsigned int lrng_state;                         //liczby losowe
    lrng_state = rng_state[idx];
    x = x_i[idx]; 
    for (k=0;k<sps;k++){{
        //x = x_i[idx];
      //  lrng_state = rng_state[idx];
        if(x>-1.0)
        {{
            //Reakcja lub pochlanianie
           
            if (rng_uni(&lrng_state)<=k_react(x)*{dt}f)//(kreacta(x, lrng_state))
            {{
                x = -1.0;
            }}
           else
            {{
                //Dyfuzja
                n1 = rng_uni(&lrng_state);
		        n2 = rng_uni(&lrng_state);
		        bm_trans(n1, n2);         //rozklad normalny
                
                x1 =fabsf( x + SD(x) * n1);
                if (x1>=({L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f+{L[4]}f))                    //dla x1 rowniez nalezy zastosowac warunek brzegowy
                {{
                    x=-1.0;
                }}
                else
                {{
                    x = fabsf(x + SD(x1)* n1 + V(x));
                
                if (x<={L[0]}f||x>({L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f+{L[4]}f))
                {{
                    x=-1.0;                                                           //(1)  oproznienie kontenera i warunek brzegowy dla x 
                                                                                      //gdy tracer po dyfuzji nadal znajduje sie w kontenerze 
                                                                                      //lub sietam pojawil, zostaje deaktywowany, aby zapewnic
                                                                                      //pusty obszar dla generacji tracerow 
                                                                                      //wowczas nie ma koniecznosci liczenia ile tracerow znajduje 
                                                                                      //sie w zadanym obszarze i ile nalezaloby uzupelnic, z gory 
                                                                                      //znana jest liczba trecerow do aktywacji     
                }}
            }}
}}
          //  x_i[idx] = x;
            
            
        }}
        
        
        if (x==-1)
        {{
            indexes[idx+({distance}-1)-(idx%{distance})]=idx;                          //w tym polu znajduje sie indeks komorki tablicy w ktorej jest -1
                                                                                     //nie wazne ktoremu watkowi uda sie zapisac jako ostatniemu, wazne jest by bylo tam -1
                                                                                     //zamiast szukac zagladamy do tej wartosci w tabeli i mamy id niezywej czastki
                                                                                     //moze da sie to zrobic jeszcze inaczej
        }}
        __syncthreads();   
   
if(idx==indexes[idx+({distance}-1)-(idx%{distance})])
        {{
x=rng_uni(&lrng_state)*{L[0]}f;


        }}
             
        __syncthreads(); 
     //   rng_state[idx] = lrng_state;
    }}
x_i[idx] = x;
}}





""".format(**pars)
    mod = SourceModule(rng_src+src,options=["--use_fast_math"]  )
    #print src
    advance_tex = mod.get_function("advance_tex")
   # advance_tex1 = mod.get_function("advance_tex1")
    return mod,advance_tex#,advance_tex1

# <codecell>



# <codecell>


# <headingcell level=3>

# Inicjalizacja kerneli i przygotowanie symulacji

# <codecell>

blocks = 2500      #ilosc blokow odpowiada ilosci tracerow w obszarze o stalym stezeniu, a kazdym bloku symulowany jest zawsze 1 tracer z tego obszaru.(2)
                   #w kazdym watku jest zagwarantowane (1) ze po kazdym kroku symulacji kontener jest oprozniony
                   #moĹĽna zwiekszyc ilosc tracerow nie zmieniajac ilosci blokow przez ustawienie parametru distance na block_size/n-> n tracerow w kazdym bloku 
block_size = 256   #dobrany aby calkowita ilosc tracerow w ciagu calej symulacji byla odpowiednia


n_tracers = blocks*block_size  

xinit = np.ones(n_tracers,dtype=np.float32)
xinit = xinit * (-1)
#polozenie -1 znaczy, ze dany watek nie jest jeszcze wykorzystywany w symulacji, ale jest gotowy do wykorzystania, 
#dzieki temu symulacja nie musi byc sterowana w nadzednym watku, aby kazdorazowo zmieniac ilosc tracerow i odpowiadajacych im watkow
distance=block_size
ainit =-1* np.ones(n_tracers, dtype=np.int32)
for i in range(0, n_tracers, distance):
    xinit[i] = (L[0]) * np.random.random_sample()


tracers= ((0 < xinit) & (xinit <= L[0])).sum()
print "  tracers" , tracers

pars = {'dt':dt, 'L':L, 'SD':SD, 's':sigma, 'k_react':k_react, 'V':VV, 'distance':distance}
mod,advance= get_kernel(pars)



dx_i = gpuarray.to_gpu (xinit)
rng_state = np.array(np.random.randint(1,2147483648,size=n_tracers),dtype=np.uint32)
rng_state_g = gpuarray.to_gpu(rng_state)
da_i = gpuarray.to_gpu (ainit)
print n_tracers,n_tracers*100000/1e9,"G","dt=",dt

# <markdowncell>

# Uruchomienie symulacji
# -----------------------
# Co x krokĂłw kontrola koncentracji w kontenerze

# <codecell>

import time
test = np.int32(2)
sps = np.int32(3000000)#2000000 950000
sps2 = np.int32(5000)#2000000 950000
size = np.int32(n_tracers)
                                   
start = time.time()
for k in range(1):
    #advance1(dx_i,da_i,rng_state_g, sps2, block=(block_size,1,1), grid=(blocks,1))
    advance(dx_i,da_i,rng_state_g, sps, block=(block_size,1,1), grid=(blocks,1))
    ctx.synchronize()
    #print ((0 < dx_i.get()) & (dx_i.get() <= L[0])).sum()
   # if k%1==0:
    #    with open('test1.log', 'w') as f:
     #       f.writelines(str(j)  + '\n' for j in dx_i.get())
elapsed = (time.time() - start)  
with open('test1.log', 'w') as f:
    f.writelines(str(j)  + '\n' for j in dx_i.get())



print elapsed,np.nonzero(np.isnan(dx_i.get()))[0].shape[0]


ctx.pop()
ctx.detach()

