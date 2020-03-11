#ifndef _KERNEL_H
#define _KERNEL_H

#include "structs.h"
#include "lock.h"

__global__ void runDev(int* clocks, Lock* locks, int* antGrid, int* objectGrid, ant_t* antArray, object_t* objectArray, double* objectDist, int dim, int numAnts, int numObjects, int radius);

#endif
