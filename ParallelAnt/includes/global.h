#ifndef _GLOBAL_VARIABLE_H
#define _GLOBAL_VARIABLE_H

// head files
#include <structs.h>

// global head files and packages
// c headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

// cpp headers
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <climits>

using namespace std;

// global variables
// parameters
extern string fileName; // input file name
extern int dim; // grid size
extern int numAnts;
extern int numObjects;
extern int numIter; // number of iterations
extern int interval; // intermid interval

// common
extern int* objectGrid;
extern int* antGrid;
extern object_t* objectArray; // location of objects
extern ant_t* antArray; // location, objId and seed of ants

// data file
extern objectInfo_t* objInfo; // object information from plain
extern int featureLength; // used to compute the distances

// graph file
extern int* adjMatrix;
extern int numEdge;

// algorithm parameters
extern int s; // radius
extern double* dist; // distance matrix
extern double alpha;
extern double k1;
extern double k2;
extern omp_lock_t* lockGrid;

// information retrieval
extern int* membership; // cluster membership
extern int* kcentroids;
extern int k;

extern double modularity;
extern double interDist;
extern double intraDist;

extern double start_time;
extern double stop_time;

// define
#define UNDEFINED -1

// random
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

#endif