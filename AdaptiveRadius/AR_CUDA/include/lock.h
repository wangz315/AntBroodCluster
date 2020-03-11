#ifndef _LOCK_H
#define _LOCK_H


struct Lock
{
	int* mutex;

	Lock()
	{
		int state = 0;

		cudaMalloc((void**) &mutex, sizeof(int));
		cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
	}

	__host__ __device__ ~Lock()
	{
		//cudaFree(mutex);
	}

	__device__ void lock()
	{
		while(atomicCAS(mutex, 0, 1) != 0);
	}

	__device__ void unlock()
	{
		atomicExch(mutex, 0);
	}
};


#endif
