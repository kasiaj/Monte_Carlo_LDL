#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265358979f
//nazwy       = [  'container','endothel' , 'intima'];
//dobre

float SDx(double x, double *SD, double *L)
{
    if(x<L[0])
    {
        return SD[0];
    }
    else if (x<L[0]+L[1]){
        return SD[1];
    }
    else if (x<L[0]+L[1]+L[2]){
        return SD[2];
    }
    else if (x<L[0]+L[1]+L[2]+L[3]){
        return SD[3];
    }
    else
    {
        return SD[4];
    }
}

 float Vx(double x, double *V, double *L)//velsigma
{
    if(x<L[0])
    {
        return V[0];
    }
    else if (x<L[0]+L[1]){
        return V[1];
    }
    else if (x<L[0]+L[1]+L[2]){
        return V[2];
    }
    else if (x<L[0]+L[1]+L[2]+L[3]){
        return V[3];
    }
    else
    {
        return V[4];
    }
}

 float k_reactx(double x, double *k_react, double *L)
{
    if (x<L[0]+L[1]+L[2]+L[3])
    {
      return k_react[4];
    }
    else
    {
      return -1.0;
    }
}


double  bm_trans(double u1, double u2)
{
	float r = sqrt(-2.0 * log(u1));
	float phi = 2.0 *  PI* u2;
	u1 = r * cos(phi);
	u2 = r * sin(phi);
	return u1;
}


float step(double x, double dt, double *L, double *SD, double *k_react, double *V)
{
  //  print "x przed" ,x
    float c = (float)rand()/(float)RAND_MAX;
    if(c<=k_reactx(x, k_react, L)*dt)
    {
        x = -1.0;
    }
    else
{
        //Dyfuzja
        float n1 = (float)rand()/(float)RAND_MAX;
        float n2 = (float)rand()/(float)RAND_MAX;
        n1 =bm_trans(n1, n2);         //rozklad normalny
               
        double x1 =fabs( x + SDx(x, SD, L) * n1);
        printf ("xiniti_trakcie= %lf\n", x1); 
if (x1>=(L[0]+L[1]+L[2]+L[3]+L[4]))                    //dla x1 rowniez nalezy zastosowac warunek brzegowy
        {
            x=-1.0;
	 }
        else
	 {
            x = fabs(x + SDx(x1, SD, L)* n1 + Vx(x, V,L));
            
           
            if (x<=L[0] || x>(L[0]+L[1]+L[2]+L[3]+L[4]))
 	     {
                x=-1.0;
		 }
	}
}
        return x;
}



int main() 
{
double D[5]         ={   6e-11     ,  6e-11  , 5.4e-6 , 3.18e-9  ,  5e-8} ;
double V[5]           = {   2.6783499775030101e-5    ,  2.6783499775030101e-5    , 2.6783499775030101e-5  , 2.6783499775030101e-5, 2.6783499775030101e-5 };
double sigma[5]       = {  0.93108378484950194,     0.93108378484950194, 0.8272, 0.9827, 0.8836};
double L[5]           = {    0.2      ,  1.8        ,10. ,2.0, 200.};
double k_react[5]     = {    0.       ,  0.        , 0. , 0., 1.4e-4};
double SD[5];
double VV[5];
double dt=0.01;
int i;
//Change units to um
for (i =0; i<5; i++)
{

	SD[i]=sqrt(D[i]*dt*2.0*1e6) ;
	VV[i]=dt*(1-sigma[i])*V[i]*1e3;
 	printf ("SDi= %lf\n", SD[i]);
}
int tracers = 2500;
int block_size = 256;  
int n_tracers = tracers*block_size;
int zarodek;
zarodek= time(NULL);
srand(zarodek);
double xinit[n_tracers];
double x;
for (i =0; i<tracers; i++)
{
	xinit[i]=(float)rand()/(float)RAND_MAX*L[0];
	printf ("xiniti= %lf\n", xinit[i]);
}
int j,k, reservior, m;
for (k=0; k<300;k++)
{
    for (j=0;j<n_tracers;j++)
    {
          x = xinit[j];
          if (x>-1.0)
	   {
		printf ("xiniti_0= %lf\n", xinit[j]);

              xinit[j] = step(x, dt, L, SD, k_react, V);
		printf ("xiniti_1= %lf\n", xinit[j]);

	   }
     }
     reservior=tracers;
     m=0;
     while (reservior>0)
     {
	   if (xinit[m]==-1.0)
	   {
		xinit[m]=(float)rand()/(float)RAND_MAX*L[0];
		reservior--;
	   }
	   m++;
//printf ("m= %i\n", m);

     }

}
FILE *pliczek;
pliczek = fopen("test2.log","w");
for(i = 0; i < n_tracers; i++)
{
fprintf(pliczek , "%f\n", xinit[i]);
}

fclose(pliczek );






return 0;
}
