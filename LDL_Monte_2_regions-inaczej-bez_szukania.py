

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
#D           = [   6e-11     ,  6e-11  , 5.4e-6  , 3.18e-9  ]
#V           = [   5.6708144518415195e-5    ,  5.6708144518415195e-5    , 5.6708144518415195e-5   , 5.6708144518415195e-5  ]
#sigma       = [  0.82148658915977035,    0.82148658915977035, 0.8272, 0.9827 ]
L           = [    0.2      ,  1.8        , 10.   , 2.  ]
k_react     = [    0.       ,  0.        , 0. , 0.]
D           = [   6e-11  ,  6e-11     , 5.4e-8  , 3.18e-8 ]#, 5e-8   ]
V           = [0.026533788267959334, 0.026533788267959334, 0.026533788267959334, 0.026533788267959334]# , 2.6783499775030101e-5  ]#, 2.2e-5  , 2.2e-5 ]
sigma       = [  0.93108378484950194,     0.93108378484950194, 0.8272, 0.9827]#0.9827, 0.8836] 


#Change units to um
D = [D_*1e6 for D_ in D]
#V = [V_*1e3 for V_ in V]
dt=1.0#0.001


cuda.init()
device = cuda.Device(0)
ctx = device.make_context()
print device.name(), device.compute_capability(),device.total_memory()/1024.**3,"GB"
print "a tak wogóle to mamy tu:",cuda.Device.count(), " urządzenia"

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
//texture<float, 2> img_fx;
//texture<float, 2> img_fy;



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

  }}
  


  __device__  float k_react(float x)
  {{
    if (x>{L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f)
    {{
      return {k_react[1]}f;
      }}
    else
    {{
      return -1.0;
      }}
  }}
  
    
    
__global__ void advance_tex(float *x_i,int *alive, int *it_i,unsigned int *rng_state,int sps)           
    {{
        int idx = blockDim.x*blockIdx.x + threadIdx.x;
        float x, x1, x0;     //x1 to delta Y z publikacji
        int k,it;       
        float n1, n2, n3,c1; 	//tez do losowania 
        unsigned int lrng_state;   //liczby losowe
        lrng_state = rng_state[idx];   //
        
        x = x_i[idx];
        it = it_i[idx];

if (it>=0) {{
    for (k=0;k<sps;k++){{//sps
        x = x_i[idx];
        x0 = x;

        if(x>-1)
        {{
            //Reakcja lub pochlanianie
            c1 = rng_uni(&lrng_state);
            if(c1<=k_react(x)*{dt}f|| x>({L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f))
            {{
                x = -1.0;
            }}
            if(x>-1.0)
            {{
                //Dyfuzja
                zbiornik = (x<{L[0]}f && x>=0);
                zbiornik1 = (x>={L[0]}f);
                n1 = rng_uni(&lrng_state);
		        n2 = rng_uni(&lrng_state);
                n3 = rng_uni(&lrng_state);
		        bm_trans(n1, n2);         //rozklad normalny
                x1 =fabsf( x + sqrtf({dt}f * D(x)  * 2.0f) * n1);
                x = fabsf(x + sqrtf({dt}f  *D(x1) * 2.0f) * n1 + {dt}f*(1-sigma(x))*V(x));// //
                if (x<={L[0]}f||x>({L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f))
                {{
                  x=-1.0;
                  }}
                            }}
          
        }}
        if (x==-1)
        {{
          alive[idx+({tracers}-1)-(idx%{tracers})]=idx; //w alive znajduje sie numer czastki ktora ma x=-1 (czyli jest jeszcze nieaktywna w symulacji) 
          }}
    
 x_i[idx] = x;
__syncthreads();   
 //wypelnienie zbiornia, robi to tylko 2622 watkow, czyli 1 na blok
    
   if(idx%{tracers}=={tracers}-1)
   {{
        x_i[alive[idx]]=rng_uni(&lrng_state)*{L[0]}f;
     }}
             
        __syncthreads(); 
        rng_state[idx] = lrng_state;
    }}
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
blocks = 2622#ilosc blokow=ilosc czastek w kontenerze o stalej gestosci
block_size = 300#264

n_tracers = blocks*block_size   
#testinit =  np.arange(n_tracers,dtype=np.int32)
xinit = np.ones(n_tracers,dtype=np.float32)
xinit = xinit * (-1)

ainit =-1* np.ones(n_tracers, dtype=np.int32)#zeros
#ginit = np.ones(n_tracers, dtype=np.bool)
for i in range(0, n_tracers, 300):
    xinit[i] = (L[0]) * np.random.random_sample() #polozenia poczatkowe -10 % w kontenerze+L[1]+L[1]+L[2]+L[3]+L[4]
   # ainit[i] = True


it = np.arange(n_tracers,dtype=np.int32)  #Numery iteracji potem zmienic na zeros
tracers= ((0 < xinit) & (xinit <= L[0])).sum()
print "  tracers" , tracers
distance=300
pars = {'dt':dt, 'L':L, 'D':D, 's':sigma, 'k_react':k_react, 'V':V, 'tracers':distance}#, 'sps': sps}
mod,advance = get_kernel(pars)


#it = np.random.randint(-1,1,n_tracers).astype(np.int32)
print xinit.max()
#print xinit.min()

dx_i = gpuarray.to_gpu (xinit)
da_i = gpuarray.to_gpu (ainit)
dit = gpuarray.to_gpu (it)
dg_i =  gpuarray.to_gpu (ginit)
#print da_i.get()
rng_state = np.array(np.random.randint(1,2147483648,size=n_tracers),dtype=np.uint32)
rng_state_g = gpuarray.to_gpu(rng_state)
print it

print n_tracers,Nsteps,n_tracers*Nsteps/1e9,"G","dt=",dt

# <markdowncell>

# Uruchomienie symulacji
# -----------------------
# Co x kroków kontrola koncentracji w kontenerze

# <codecell>

import time
test = np.int32(0)
sps = np.int32(20000)
size = np.int32(n_tracers)
                                          #(1)
                                            #(2)

print (Nsteps/2/sps)
start = time.time()
for k in range(1500):#(Nsteps/2/sps):80000
    advance(dx_i,da_i,dit,rng_state_g, sps, block=(block_size,1,1), grid=(blocks,1))#,dg_i, size
    #print ((0 < dx_i.get()) & (dx_i.get() <= L[0])).sum()
    if k%100==50:
        print da_i.get()[299]
        with open('test2.log', 'w') as f:
            f.writelines(str(j)  + '\n' for j in dx_i.get())


ctx.synchronize()
elapsed = (time.time() - start)  
print elapsed,np.nonzero(np.isnan(dx_i.get()))[0].shape[0]

# <codecell>



ctx.pop()
ctx.detach()




