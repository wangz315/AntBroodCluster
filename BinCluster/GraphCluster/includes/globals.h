#ifndef _GLOBAL_VARIABLE_H
#define _GLOBAL_VARIABLE_H

// head files

// global head files and packages
// c headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
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
// args
extern int argLength;
extern char** argArray;

// parameters
extern char* fileName; // input file name
extern int dim; // grid size
extern int numAnts;
extern int numObjects;
extern int numEdge;
extern int numIter; // number of iterations
extern int interval; // intermid interval
extern int* adjMatrix;


extern int* antArray;
extern int* objectArray;
extern int* membership;

// algorithm parameters
extern int s; // radius
extern double alpha;
extern double k1;
extern double k2;


extern int k;


// random
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

#endif