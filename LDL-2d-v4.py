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



# <codecell>

import numpy as np 
from scipy.sparse import dia_matrix
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as pl
import pycuda.gpuarray as gpuarray
from pycuda.curandom import rand as curand

from pycuda.compiler import SourceModule
import pycuda.driver as cuda

# <markdowncell>
print "po import"
# Parametry:

# <codecell>

nazwy       = [  'container','endothel' , 'intima']
#dobre
D           =[   6e-11     ,  6e-11  , 5.4e-6 , 3.18e-9  ,  5e-8] 
V           = [   2.6783499775030101e-5    ,  2.6783499775030101e-5    , 2.6783499775030101e-5  , 2.6783499775030101e-5, 2.6783499775030101e-5 ]
sigma       = [  0.93108378484950194,     0.93108378484950194, 0.8272, 0.9827, 0.8836]
L           = [    0.2      ,  1.8        ,10. ,2.0, 200.]
k_react     = [    0.       ,  0.        , 0. , 0., 1.4e-4]

#nazwy       = [  'container','endothel' , 'intima', 'IEL'   ,'media']
#D           =[   6e-9     ,  6e-9  , 5.4e-6 , 3.18e-9  ,  5e-8] #[   6e-11     ,  6e-11  , 5.4e-6  , 3.18e-9  ]
#V           = [   2.6783499775030101e-5    ,  2.6783499775030101e-5    , 2.6783499775030101e-5  , 2.6783499775030101e-5, 2.6783499775030101e-5 ]#, 2.6783499775030101e-5  
#sigma       = [  0.93108378484950194,     0.93108378484950194, 0.8272, 0.9827, 0.8836]#, 0.9827]#0.9827, 0.8836] 
#L           = [    0.2      ,  1.8        ,10. ,1.,0.]#2.0, 200.]#,5.    ]#,    2.   , 200.   ]#test rozmiaru kontenera
#k_react     = [    0.       ,  0.        , 0. , 0., 0.]#1.4e-4]

K           = [   3.32e-15  , 2e-10   ,4.392e-13, 2e-12   ]
mu          = [   0.72e-3   , 0.72e-3 , 0.72e-3 , 0.72e-3 ]


#Change units to um
D = [D_*1e6 for D_ in D]
V = [V_*1e3 for V_ in V]
dt=0.01#1.0#

SD=[np.sqrt(D_*dt*2.0) for D_ in D]
sigmat =[dt*(1-sigma_)for sigma_ in sigma ] 
VVV=[dt*(1-sigma_)*V_ for sigma_, V_ in zip(sigma, V)]


#z 1d
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
VVV=[dt*(1-sigma_)*V_ for sigma_, V_ in zip(sigma, V)]

# <codecell>


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

    __device__  float SD(float x)
{{
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

__device__  float velsigma(float x, float y)
{{

  if(x>{L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f)
    {{
       return ({vvel0}f+({vvelm}f))*{s[4]}f*copysignf(1, y)*copysignf(1, y)+expf(0.001*y)-expf(0.001*y);//+exp(y)-exp(y)  ;//*copysignf(1, y)*copysignf(1, y) *exp(0.01f)/exp(0.01f)*cos(0.0000001f*y)
    }}
    else if (x>{L[0]}f+{L[1]}f+{L[2]}f){{
        return ({vvel0}f+({vvelm}f))*{s[3]}f*copysignf(1, y)*copysignf(1, y)+expf(0.001*y)-expf(0.001*y);//*exp(y)/exp(y) ;//*copysignf(1, y)*copysignf(1, y) *exp(0.01f)/exp(0.01f)*cos(0.0000001f*y)
    }}
    else if (x>{L[0]}f+{L[1]}f){{
        return ({vvel0}f+({vvelm}f))*{s[2]}f*copysignf(1, y)*copysignf(1, y)+expf(0.001*y)-expf(0.001*y);//*exp(y)/exp(y)  ;//*copysignf(1, y)copysignf(1, y) *exp(0.01f)/exp(0.01f)*cos(0.0000001f*y)
    }}
    else {{
       
       return ({vvel0}f+({vvelm}f))*{s[1]}f*copysignf(1, y)*copysignf(1, y)+expf(0.001*y)-expf(0.001*y);//*exp(y)/exp(y);//return ({vel0}f+(({velm}f)))*exp(y)/exp(y) ;//*copysignf(1, y)*copysignf(1, y) ; *exp(0.01f)/exp(0.01f)  *cos(0.0000001f*y)
    }}
 
  
}}
 /* 

__device__  float velsigma(float x, float y)
{{
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
   // else if (x>{L[0]}f){{
   //     return {V[1]}f;
   // }}
    else
    {{
        return {V[0]}f;
    }}
}}

*/

  
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
  
    
    
__global__ void advance_tex(float *x_i, float *y_i,int *indexes,unsigned int *rng_state,int sps)    
{{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    float x, x1, y, y1;                                     //x1 to delta Y z publikacji
y=0;    
int k;       
    float n1, n2; 	                             //do losowania
    unsigned int lrng_state;                         //liczby losowe
    x = x_i[idx];
    lrng_state = rng_state[idx];
 y = y_i[idx];
    for (k=0;k<sps;k++){{
       // x = x_i[idx];
        
       // lrng_state = rng_state[idx];
        if(x>-1.0)
        {{
            //Reakcja lub pochlanianie
            //c1 = rng_uni(&lrng_state);
            if(rng_uni(&lrng_state)<=k_react(x)*{dt}f)
            {{
                x = -1.0;
            }}
            else
            {{
                //Dyfuzja
                n1 = rng_uni(&lrng_state);
		        n2 = rng_uni(&lrng_state);
		       // n3 = rng_uni(&lrng_state);
		        bm_trans(n1, n2);         //rozklad normalny
                
                x1 =fabsf( x + SD(x) * n1);
                if (x1>=({L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f+{L[4]}f))                    //dla x1 rowniez nalezy zastosowac warunek brzegowy
                {{
                    x=-1.0;
                }}
                else
                {{
                //  y = y_i[idx];
                    x = fabsf(x + SD(x1) * n1 + velsigma(x, y));
                
                if (x<={L[0]}f||x>({L[0]}f+{L[1]}f+{L[2]}f+{L[3]}f+{L[4]}f))
                {{
                    x=-1.0;                                                          /* //(1)  oproznienie kontenera i warunek brzegowy dla x 
                                                                                      //gdy tracer po dyfuzji nadal znajduje sie w kontenerze 
                                                                                      //lub sietam pojawil, zostaje deaktywowany, aby zapewnic
                                                                                      //pusty obszar dla generacji tracerow 
                                                                                      //wowczas nie ma koniecznosci liczenia ile tracerow 
                                                                                      //znajduje sie w zadanym obszarze i ile nalezaloby uzupelnic, 
                                                                                      z gory znana jest liczba trecerow do aktywacji   */   
                }}
                else
                {{

                  
                  y=y + SD(x1)* n2 ;
                  
                if (y>{limit_up}f)
                {{
                    y = 2*{limit_up}f-y;
                }}  
                else if (y<{limit_down}f)
                {{
                    y=2*({limit_down}f)-y;
                }}
                 //  y_i[idx]=y1;
}}
            }}
    }}
           // x_i[idx] = x;
         
            
        }}
        
        
        if (x==-1)
        {{
            indexes[idx+({distance}-1)-(idx%{distance})]=idx;                          //w tym polu znajduje sie indeks komorki tablicy w ktorej jest -1
                                                                                     //nie wazne ktoremu watkowi uda sie zapisac jako ostatniemu, wazne jest by bylo tam -1
                                                                                     //zamiast szukac zagladamy do tej wartosci w tabeli i mamy id niezywej czastki
                                                                                     //moze da sie to zrobic jeszcze inaczej
        }}
        __syncthreads();   
    
        if(idx== indexes[idx+({distance}-1)-(idx%{distance})])
        {{

            //int iddd=indexes[idx];
            x=rng_uni(&lrng_state)*{L[0]}f;
            y=rng_uni(&lrng_state)*(fabsf({limit_down}f)+fabsf({limit_up}f))+({limit_down}f);
            //indexes[idx]=-1;
        }}
             
        __syncthreads(); 
       // rng_state[idx] = lrng_state;
    }}
y_i[idx]=y;
rng_state[idx] = lrng_state;
x_i[idx] = x;
}}
""".format(**pars)
    mod = SourceModule(rng_src+src,options=["--use_fast_math"]  )
   # print src
    advance_tex = mod.get_function("advance_tex")
    return mod,advance_tex

# <codecell>


def prepare():
    Rcell = 15e-3 # mm
    area=.64 # mm^2
    A = np.pi*Rcell**2/area
    
    w=14.3*1e-6
    r_m = 11e-6
    a = r_m/w
    
    sigma_lj = 1-(1-3/2.*a**2+0.5*a**3)*(1-1/3.*a**2)
    B=1-sigma_lj
    
    Kend_70mmHg = 3.22e-21
    phi_0=5e-4
    Klj=(w**2/3.)*(4.*w*phi_0)/Rcell * (1e-6)
    C = Kend_70mmHg - Klj
    
    d = (w**2/3.)*(4.*w)/Rcell * (1e-6)
    #print C
    
    mmHg2Pa = 133.3
    E = mmHg2Pa*70.0
    
    mu1=mu[1:]
    K1=K[1:]
    L1=L[2:]
    Ss = [L_*mu_/K_ for L_,K_,mu_ in zip(L1,K1,mu1)]
    F=sum(Ss)
    
    G = (L[0]+L[1])*mu[0]
    
    
    return A,a , B, C,d, E, F, G

# <codecell>


# <codecell>

def klj(y):
    Rcell = 15e-3 # mm
    area=.64 # mm^2
    A = np.pi*Rcell**2/area
    
    w=14.3*1e-6
    
    
    d = (w**2/3.)*(4.*w)/Rcell * (1e-6)
    
    Wmax = 50.
    Wmin = 40.
    srodek = (Wmax+Wmin)/2
    szerokosc = srodek - Wmin
    print srodek, szerokosc
    WSS=srodek+szerokosc*np.sign(y)
    
   # WSS=10.0-9.9*exp(-y*y/(2*5*5));
    SI = 0.38*np.exp(-0.79*WSS) + 0.225*np.exp(-0.043*WSS);
    MC = 0.003797* np.exp(14.75*SI); 
    LC = 0.307 + 0.805 * MC ;
    phi = LC*A;
    return d *phi;

# <codecell>

def VV(y):
    
    Rcell = 15e-3 # mm
    w=14.3*1e-6
    Kend_70mmHg = 3.22e-21
    phi_0=5e-4
    Klj=(w**2/3.)*(4.*w*phi_0)/Rcell * (1e-6)
    C = Kend_70mmHg - Klj
    mu1=mu[1:]
    K1=K[1:]
    L1=L[2:]
    Ss = [L_*mu_/K_ for L_,K_,mu_ in zip(L1,K1,mu1)]
    F=sum(Ss)
    
    G = (L[0]+L[1])*mu[0]
    Kend = C+klj(y);
    RW0 = G/(Kend*1e6);
    RWC = RW0+F;
    
    mmHg2Pa = 133.3
    E = mmHg2Pa*70.0
    return E/RWC*1e6;

# <codecell>

def sig(y):
    w=14.3*1e-6
    r_m = 11e-6
    a = r_m/w
    Rcell = 15e-3 # mm
    sigma_lj = 1-(1-3/2.*a**2+0.5*a**3)*(1-1/3.*a**2)
    b=1-sigma_lj
    Kend_70mmHg = 3.22e-21
    phi_0=5e-4
    Klj=(w**2/3.)*(4.*w*phi_0)/Rcell * (1e-6)
    c = Kend_70mmHg - Klj
    return 1-(b*klj(y))/(c+klj(y));

# <codecell>

print VV(0)

# <headingcell level=3>

# Inicjalizacja kerneli i przygotowanie symulacji

# <codecell>

blocks =2500      #ilosc blokow odpowiada ilosci tracerow w obszarze o stalym stezeniu, a kazdym bloku symulowany jest zawsze 1 tracer z tego obszaru.(2)
                   #w kazdym watku jest zagwarantowane (1) ze po kazdym kroku symulacji kontener jest oprozniony  50000
block_size = 256   #dobrany aby calkowita ilosc tracerow w ciagu calej symulacji byla odpowiednia


n_tracers = blocks*block_size   #polowa to martwe dusze

xinit = np.ones(n_tracers,dtype=np.float32)
xinit = xinit * (-1)
yinit = np.ones(n_tracers,dtype=np.float32)
#polozenie -1 znaczy, ze dany watek nie jest jeszcze wykorzystywany w symulacji, ale jest gotowy do wykorzystania,
#dzieki temu symulacja nie musi byc sterowana w nadzednym watku, aby kazdorazowo zmieniac ilosc tracerow i odpowiadajacych im watkow
distance=block_size#/2
ainit =-1* np.ones(n_tracers, dtype=np.int32)
for i in range(0, n_tracers, distance):
    xinit[i] = (L[0]) * np.random.random_sample()
    yinit[i] = -500.0+1000.0*np.random.random_sample()
#Z pliku
#xinit = np.ones(n_tracers,dtype=np.float32)
#binit= np.genfromtxt("testx_5norm18",dtype=np.float32)
#xinit=xinit*binit
#yinit = np.ones(n_tracers,dtype=np.float32)
#binit= np.genfromtxt("testy_5norm18",dtype=np.float32)
#yinit=yinit*binit
print D
tracers= ((0 < xinit) & (xinit <= L[0])).sum()
print "  tracers" , tracers


lmax = 500.0
lmin=-500.0
#parametry do skoku
smin = sig(lmin)
smax = sig(lmax)
s0=sigma[0]#(smin+smax)/2#srodek
sm = 0.0001*sigma[0]# s0-smin#szerokosc
vmin=VV(lmin)
vmax = VV(lmax)
v0=V[0]#(vmin+vmax)/2#srodek    velocity 
vm = 0.0001*V[0]#v0-vmin#szerokosc

smin = s0-sm
smax = s0+sm
vmin=v0-vm
vmax = v0+vm
print v0, vm, s0, sm
#print V
vvmin=(1-smin)*vmin*dt
vvmax=(1-smax)*vmax*dt
vv0=(vvmin+vvmax)/2#srodek    velocity*(1-sigma)*dt
vvm = vv0-vvmin#szerokosc
print vv0, vvm
pars = {'dt':dt, 'L':L, 'SD':SD, 's':sigmat, 'k_react':k_react, 'V':VVV, 'distance':distance,  'limit_up':lmax, 'limit_down':lmin,  'vel0':vv0, 'velm':vvm, 'vvel0':v0, 'vvelm':vm}


mod,advance = get_kernel(pars)



dx_i = gpuarray.to_gpu (xinit)
dy_i = gpuarray.to_gpu (yinit)
rng_state = np.array(np.random.randint(1,2147483648,size=n_tracers),dtype=np.uint32)
rng_state_g = gpuarray.to_gpu(rng_state)
da_i = gpuarray.to_gpu (ainit)
print n_tracers,n_tracers*5000/1e9,"G","dt=",dt

# <markdowncell>

# Uruchomienie symulacji
# -----------------------
# Co x krokĂłw kontrola koncentracji w kontenerze

# <codecell>

import time
test = np.int32(0)
sps = np.int32(1000000)#2000000 950000
size = np.int32(n_tracers)
                          
start = time.time()
for k in range(3):
    advance(dx_i,dy_i,da_i,rng_state_g, sps, block=(block_size,1,1), grid=(blocks,1))
    ctx.synchronize()
    filex = "x_2d_wd_step"
    filey="y_2d_wd_step"
    #x_file = "fit_x_5050_"+str(k)+".jpg"
    #y_file = "fit_y_5050_"+str(k)+".jpg"
    #hist_file = "fit_hist_5050_"+str(k)+".jpg"
    #print filex
    #print ((0 < dx_i.get()) & (dx_i.get() <= L[0])).sum()
    x=dx_i.get()
    y=dy_i.get()
    #y=y[x>-1.0]
    #x=x[x>-1.0]
    if k%1==0:
        with open(filex, 'w') as f:
            f.writelines(str(j)  + '\n' for j in x)
        with open(filey, 'w') as f:
            f.writelines(str(j)  + '\n' for j in y)


elapsed = (time.time() - start)  
print elapsed,np.nonzero(np.isnan(dx_i.get()))[0].shape[0]

# <codecell>

print 2500*256*3000000/elapsed/1e9

# <codecell>
ctx.pop()
ctx.detach()
