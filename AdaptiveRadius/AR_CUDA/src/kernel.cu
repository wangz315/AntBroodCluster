#include "kernels.h"


#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876


// prototypes
__device__ double atomicAddDouble(double* address, double val);
__device__ double randCuda(long *idum);
__device__ void move(int* clocks, ant_t* antArray, int dim);
__device__ double computeProbabilityOriginal(Lock* locks, int objectI, int x, int y, int radius, int dim, int numObjects, int* objectGrid, double* objectDist);
__device__ double computeProbability(Lock* locks, int objectI, int x, int y, int radius, int dim, int numObjects, int* objectGrid, double* objectDist);
__device__ void antRunDevOriginal(Lock* locks, int* antGrid, int* objectGrid, ant_t* antArray, object_t* objectArray, double* objectDist, int dim, int numAnts, int numObjects, int radius);
__device__ void antRunDev(Lock* locks, int* antGrid, int* objectGrid, ant_t* antArray, object_t* objectArray, double* objectDist, int dim, int numAnts, int numObjects, int radius);


// functions
__device__ double atomicAddDouble(double* address, double val)
{
	unsigned long long int* address_as_ull = 
	    (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__device__ double randCuda(long *idum)
{
	int k;
	double ans;

	k =(*idum)/IQ;
	*idum = IA * (*idum - k * IQ) - IR * k;
	if (*idum < 0 ) *idum += IM;
	ans = AM * (*idum);
	return ans;
}

__device__ void move(int* clocks, Lock* locks, int* antGrid, ant_t* antArray, int dim)
{
	
	int factor = 1;
	int randNum = randCuda(&antArray[blockIdx.x].seed) * ((factor*2+1)*(factor*2+1)-1);// *(max-1), to avoid move to same location

	//increase 2nd half to avoid move to same location
	if(randNum >= ((factor*2+1)*(factor*2+1)/2))
	{
		randNum++;
	}

	int x = randNum % (factor*2+1) - factor + antArray[blockIdx.x].x;
	int y = randNum / (factor*2+1) - factor + antArray[blockIdx.x].y;

	x %= dim;
	y %= dim;

	if(x < 0)
	{
		x += dim;
	}
	
	if(y < 0)
	{
		y += dim;
	}	
		
	locks[x*dim+y].lock();
	if(antGrid[x*dim+y] == -1)
	{
		antGrid[antArray[blockIdx.x].x*dim+antArray[blockIdx.x].y] = -1;
		antArray[blockIdx.x].x = x;
		antArray[blockIdx.x].y = y;
		antGrid[x*dim+y] = blockIdx.x;
	}
	locks[x*dim+y].unlock();

}


__device__ double computeProbability(Lock* locks, int objectI, int x, int y, int radius, int dim, int numObjects, int* objectGrid, double* objectDist)
{
	int objectJ;
	int xi, yj;
	int xs = x - radius;
	int ys = y - radius;
	double sim = 0.0;
	double total = 0.0;

	for(int i = 0; i < radius*2+1; i++)
	{
		xi = (xs + i) % dim;

		for(int j = 0; j < radius*2+1; j++)
		{
			if (j != x && i != y)
			{
				yj = (ys + j) % dim;

				if(xi < 0)
				{
					xi += dim;
				}

				if(yj < 0)
				{
					yj += dim;
				}

				objectJ = objectGrid[xi*dim+yj];
				if(objectJ > -1)
				{
					sim = objectDist[objectI*numObjects+objectJ];
					total += exp(-1*sim);
				}
			}
		}
	}

	int area = (powf((radius*2) + 1, 2) - 1);
	double density = total / area;
	density = max(min(density, 1.0), 0.0);
	double temp = exp(-1 * powf(density, 2));
	return  (1 - temp) / (1 + temp);
}


__device__ void antRunDev(int* clocks, Lock* locks, int* antGrid, int* objectGrid, ant_t* antArray, object_t* objectArray, double* objectDist, int dim, int numAnts, int numObjects, int radius)
{
	int x = antArray[blockIdx.x].x;
	int y = antArray[blockIdx.x].y;
  

	if(objectGrid[x*dim+y] > -1)
	{
		if(antArray[blockIdx.x].objectId == -1)
		{
			double Ppick = 1 - computeProbability(locks, antArray[blockIdx.x].objectId, x, y, radius, dim, numObjects, objectGrid, objectDist);


			if(Ppick > randCuda(&antArray[blockIdx.x].seed))
			{
				antArray[blockIdx.x].objectId = objectGrid[x*dim+y];
				objectGrid[x*dim+y] = -1;

			}
			else
			{
				move(clocks, locks, antGrid, antArray, dim);
			}
		}
		else
		{
			move(clocks, locks, antGrid, antArray, dim);
		}
	}
	else
	{
		if(antArray[blockIdx.x].objectId > -1)
		{
			double Pdrop = computeProbability(locks, antArray[blockIdx.x].objectId, x, y, radius, dim, numObjects, objectGrid, objectDist);

			if(Pdrop > randCuda(&antArray[blockIdx.x].seed))
			{
				objectGrid[x*dim+y] = antArray[blockIdx.x].objectId;
				antArray[blockIdx.x].objectId = -1;
			}
			else
			{
				move(clocks, locks, antGrid, antArray, dim);
			}
		}
		else
		{
			move(clocks, locks, antGrid, antArray, dim);
		}
	}
}


__global__ void runDev(int* clocks, Lock* locks,int* antGrid, int* objectGrid, ant_t* antArray, object_t* objectArray, double* objectDist, int dim, int numAnts, int numObjects, int radius)
{	
	for(int i = 0; i < 1000; i++)
	{
		antRunDev(clocks, locks, antGrid, objectGrid, antArray, objectArray, objectDist, dim, numAnts, numObjects, radius);	
	}
}












