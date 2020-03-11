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


class Node
{
public:
	Node();
	~Node();

	int objectId;
	Node* next;
};


class Bin
{
public:
	Bin();
	~Bin();

	// access to an random object in this bin
	int rand_obj();
	// insert
	void insert(int id);
	// delete
	void remove(int id);

	int size;
	int id;
	Node* head;
	Bin* next;
};


class Brood
{
public:
	Brood(int n);
	~Brood();

	// create an new empty bin, return bin id
	int create_bin();
	// remove bin by id
	void delete_bin(int id);
	// get bin by id
	Bin* get_bin(int id);
	// access to an random bin
	Bin* rand_bin();
	// print
	void print();

	int size;
	int idCount;
	Bin* head;
};

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
extern double* features;
extern int featureLength;

extern int* antArray;
extern int* objectArray;
extern Brood* brood;
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