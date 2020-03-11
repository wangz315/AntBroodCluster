#include "structs.h"
#include "kernels.h"
#include "timer.h"
#include "lock.h"

#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <iomanip>


using namespace std;

// variables
string colors[11] = {"\033[0m", "\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[35m", "\033[36m", "\033[37m", "\033[38m", "\033[39m", "\033[40m"};

// prototypes
void readObjectFileArtificial(string filePath, objectInit_t** objInit, int* numObjects, int* numAnts, int* featureLength, int* dim);
void readObjectFileIris(string filePath, objectInit_t** objInit, int* numObjects, int* numAnts, int* featureLength, int* dim);
void readObjectFileYest(string filePath, objectInit_t** objInit, int* numObjects, int* numAnts, int* featureLength, int* dim);
double calculateDistance(double* row1, double* row2, int length);
void calculateAllDistance(objectInit_t* objInit, double* objectDist, int numObjects, int featureLength);
void initializeGrid(int* objectGrid, int dim);
void printGrid(int* objectGrid, int dim);
void printGridCluster(int* grid, objectInit_t* objInit, int dim);
void printGridClusterBlack(int* grid, objectInit_t* objInit, int dim);
void randomDrop(object_t* objectArray, int numObjects, int* objectGrid, int dim);
void assignAnt(int* antGrid, object_t* objectArray, ant_t* antArray, int numAnts, int dim);
void check_error(cudaError_t status, const char *msg);


void readObjectFileArtificial(string filePath, objectInit_t** objInit, int* numObjects, int* numAnts, int* featureLength, int* dim)
{
	ifstream fin;
	string str;
	fin.open(filePath.c_str());

	*numObjects = 250;
	*numAnts = 25;
	*featureLength = 10;
	*dim = 35;

	*objInit = (objectInit_t*) malloc(*numObjects * sizeof(objectInit_t));

	for(int i = 0; i < *numObjects; i++)
	{
		(*objInit)[i].features = (double*) malloc(*featureLength * sizeof(double));
	}
  
	for(int i = 0; i < *numObjects; i++)
	{
		getline(fin, str);
		sscanf(str.c_str(), "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", &(*objInit)[i].clusterId, &(*objInit)[i].features[0], &(*objInit)[i].features[1], &(*objInit)[i].features[2], 
			&(*objInit)[i].features[3], &(*objInit)[i].features[4], &(*objInit)[i].features[5], &(*objInit)[i].features[6], &(*objInit)[i].features[7], &(*objInit)[i].features[8], &(*objInit)[i].features[9]);

	//cout << setprecision(11) << objInit[i].features[0] << ",";
	}

	fin.close();
}


void readObjectFileIris(string filePath, objectInit_t** objInit, int* numObjects, int* numAnts, int* featureLength, int* dim)
{
	ifstream fin;
	string str;
	fin.open(filePath.c_str());

	*numObjects = 150;
	*numAnts = 15;
	*featureLength = 4;
	*dim = 25;

	*objInit = (objectInit_t*) malloc(*numObjects * sizeof(objectInit_t));

	for(int i = 0; i < *numObjects; i++)
	{
		(*objInit)[i].features = (double*) malloc(*featureLength * sizeof(double));
	}
  
	for(int i = 0; i < *numObjects; i++)
	{
		getline(fin, str);
		sscanf(str.c_str(), "%lf,%lf,%lf,%lf,%d\n", &(*objInit)[i].features[0], &(*objInit)[i].features[1], &(*objInit)[i].features[2], 
			&(*objInit)[i].features[3], &(*objInit)[i].clusterId);

	//cout << setprecision(11) << objInit[i].features[0] << ",";
	}

	fin.close();
}


void readObjectFileYest(string filePath, objectInit_t** objInit, int* numObjects, int* numAnts, int* featureLength, int* dim)
{
	ifstream fin;
	string str;
	char* objectName = (char*) malloc(10*sizeof(char));
	fin.open(filePath.c_str());

	*numObjects = 1484;
	*numAnts = 148;
	*featureLength = 8;
	*dim = 200;

	*objInit = (objectInit_t*) malloc(*numObjects * sizeof(objectInit_t));

	for(int i = 0; i < *numObjects; i++)
	{
		(*objInit)[i].features = (double*) malloc(*featureLength * sizeof(double));
	}
  
	for(int i = 0; i < *numObjects; i++)
	{
		getline(fin, str);
		sscanf(str.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%s\n", &(*objInit)[i].features[0], &(*objInit)[i].features[1], &(*objInit)[i].features[2], 
			&(*objInit)[i].features[3], &(*objInit)[i].features[4], &(*objInit)[i].features[5], &(*objInit)[i].features[6], &(*objInit)[i].features[7], objectName);


		if(strcmp(objectName, "MIT") == 0)
		{
			(*objInit)[i].clusterId = 0;
		}
		else if(strcmp(objectName, "NUC") == 0)
		{
			(*objInit)[i].clusterId = 1;
		}
		else if(strcmp(objectName, "CYT") == 0)
		{
			(*objInit)[i].clusterId = 2;
		}
		else if(strcmp(objectName, "ME1") == 0)
		{
			(*objInit)[i].clusterId = 3;
		}
		else if(strcmp(objectName, "EXC") == 0)
		{
			(*objInit)[i].clusterId = 4;
		}
		else if(strcmp(objectName, "ME2") == 0)
		{
			(*objInit)[i].clusterId = 5;
		}
		else if(strcmp(objectName, "ME3") == 0)
		{
			(*objInit)[i].clusterId = 6;
		}
		else if(strcmp(objectName, "VAC") == 0)
		{
			(*objInit)[i].clusterId = 7;
		}
		else if(strcmp(objectName, "POX") == 0)
		{
			(*objInit)[i].clusterId = 8;
		}
		else// if(strcmp(objectName, "ERL") == 0)
		{
			(*objInit)[i].clusterId = 9;
		}



	}

	fin.close();
}


double calculateDistance(double* row1, double* row2, int length)
{
	double sum = 0.0;

	for(int i = 0; i < length; i++)
	{
		sum = sum + pow((row1[i] - row2[i]), 2.0);
	}
	return sqrt(sum);
}


void calculateAllDistance(objectInit_t* objInit, double* objectDist, int numObjects, int featureLength)
{
	for(int i = 0; i < numObjects; i++)
	{
		for(int j = 0; j < numObjects; j++)
		{
			if(i > j)
			{
				objectDist[i * numObjects + j] = calculateDistance(objInit[i].features, objInit[j].features, featureLength);
				objectDist[j * numObjects + i] = objectDist[i * numObjects + j];
			}
		}
	}
}

void initializeGrid(int* objectGrid, int dim)
{
	for(int i = 0; i < dim; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			objectGrid[i*dim+j] = -1;
		}
	}
}


void printGrid(int* grid, int dim)
{
	cout << endl;
	for(int i = 0; i < dim; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			printf("%4d", grid[i*dim+j] + 1);
		}
		cout << endl;
	}
	cout << endl;
}


void printGridCluster(int* grid, objectInit_t* objInit, int dim)
{
	cout << colors[0] << "====================================================================================================" << endl;
	for(int i = 0; i < dim; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			if(grid[i*dim+j] > -1)
			{
				cout << colors[objInit[grid[i*dim+j]].clusterId + 1] << setw(2) << objInit[grid[i*dim+j]].clusterId + 1;
			}
			else
			{
				cout << colors[0] << setw(2) << ' ';
			}
		}
		cout << endl;
	}
	cout << colors[0] << "====================================================================================================" << endl;
}


void printGridClusterBlack(int* grid, objectInit_t* objInit, int dim)
{
	cout <<  "====================================================================================================" << endl;
	for(int i = 0; i < dim; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			if(grid[i*dim+j] > -1)
			{
				cout << setw(2) << objInit[grid[i*dim+j]].clusterId + 1;
			}
			else
			{
				cout << setw(2) << ' ';
			}
		}
		cout << endl;
	}
	cout << "====================================================================================================" << endl;
}


void randomDrop(object_t* objectArray, int numObjects, int* objectGrid, int dim)
{
	int x, y, posTaken;

	for(int i = 0; i < numObjects; i++)
	{
		posTaken = 0;
		while(posTaken == 0)
		{
			x = rand()%dim;
			y = rand()%dim;
			if(objectGrid[x*dim+y] < 0)
			{
				posTaken = 1;
				objectGrid[x*dim+y]=i;
				objectArray[i].x = x;
				objectArray[i].y = y;
			}   
		}
	}
}


void assignAnt(int* antGrid, object_t* objectArray, ant_t* antArray, int numAnts, int dim)
{
	srand(time(NULL));
	for(int i = 0; i < numAnts; i++)
	{
		antArray[i].x = objectArray[i].x;
		antArray[i].y = objectArray[i].y;
		antArray[i].objectId = -1;
		antArray[i].seed = rand();

		antGrid[antArray[i].x*dim+antArray[i].y] = i;
	}
}


void check_error(cudaError_t status, const char *msg)
{
	if (status != cudaSuccess)
	{
		const char *errorStr = cudaGetErrorString(status);
		printf("%s:\n%s\nError Code: %d\n\n", msg, errorStr, status);
		exit(status); // bail out immediately (makes debugging easier)
	}
}



int main()
{
	cout << colors[0] << "Start." << endl;

	// parameters
	string artificialFilePath = "testFiles/ArtificialData.csv";
	string yestFilePath = "testFiles/yest.csv";
	string irisPath = "testFiles/iris.csv";
	int dim = 0;
	int numAnts = 0;
	int numObjects = 0;
	int featureLength = 0;
	Lock* locks;
	Lock* locksDev;
	objectInit_t* objInit;
	cudaError_t status;
	
	
	// init
	readObjectFileArtificial(artificialFilePath, &objInit, &numObjects, &numAnts, &featureLength, &dim);
	//readObjectFileIris(irisPath, &objInit, &numObjects, &numAnts, &featureLength, &dim);
	//readObjectFileYest(yestFilePath, &objInit, &numObjects, &numAnts, &featureLength, &dim);

	// host memory allocate
	ant_t* antArray = (ant_t*) malloc(numAnts * sizeof(ant_t));
	object_t* objectArray = (object_t*) malloc(numObjects * sizeof(object_t));
	double* objectDist = (double*) malloc(numObjects * numObjects * sizeof(double));

	int* objectGrid = (int*) malloc(dim * dim * sizeof(int));
	int* antGrid = (int*) malloc(dim* dim * sizeof(int));

	calculateAllDistance(objInit, objectDist, numObjects, featureLength);
	initializeGrid(objectGrid, dim);
	initializeGrid(antGrid, dim);
	randomDrop(objectArray, numObjects, objectGrid, dim);
	assignAnt(antGrid, objectArray, antArray, numAnts, dim);


	// cuda memory allocate and copy
	ant_t* antArrayDev;
	object_t* objectArrayDev;
	double* objectDistDev;

	int* objectGridDev;
	int* antGridDev;
	int* clockDev;

	status = cudaMalloc(&antArrayDev, numAnts * sizeof(ant_t));
	check_error(status, "Error allocating dev buffer antArray.");
	status = cudaMemcpy(antArrayDev, antArray, numAnts * sizeof(ant_t), cudaMemcpyHostToDevice);
	check_error(status, "Error memory copy for antArray.");

	status = cudaMalloc(&objectArrayDev, numObjects * sizeof(object_t));
	check_error(status, "Error allocating dev buffer objectArray.");
	status = cudaMemcpy(objectArrayDev, objectArray, numObjects * sizeof(object_t), cudaMemcpyHostToDevice);
	check_error(status, "Error memory copy for objectArray.");

	status = cudaMalloc(&objectDistDev, numObjects * numObjects * sizeof(double));
	check_error(status, "Error allocating dev buffer objectDist.");
	status = cudaMemcpy(objectDistDev, objectDist, numObjects * numObjects * sizeof(double), cudaMemcpyHostToDevice);
	check_error(status, "Error memory copy for objectDist.");

	status = cudaMalloc(&objectGridDev, dim * dim * sizeof(int));
	check_error(status, "Error allocating dev buffer objectGrid.");
	status = cudaMemcpy(objectGridDev, objectGrid, dim * dim * sizeof(int), cudaMemcpyHostToDevice);
	check_error(status, "Error memory copy for objectGrid.");

	status = cudaMalloc(&antGridDev, dim * dim * sizeof(int));
	check_error(status, "Error allocating dev buffer antGrid.");
	status = cudaMemcpy(antGridDev, antGrid, dim * dim * sizeof(int), cudaMemcpyHostToDevice);
	check_error(status, "Error memory copy for antGrid.");

	status = cudaMalloc(&clockDev, numAnts * sizeof(int));
	check_error(status, "Error allocating dev buffer clockDev.");

	int numLocks = dim * dim;

	locks = (Lock*) malloc(numLocks * sizeof(Lock));

	Lock lock;

	for(int i = 0; i < dim * dim; i++)
	{
		locks[i] = lock;
	}
	

	status = cudaMalloc(&locksDev, numLocks * sizeof(Lock));
	check_error(status, "Error allocating dev buffer locks.");
	status = cudaMemcpy(locksDev, locks, numLocks * sizeof(Lock), cudaMemcpyHostToDevice);
	check_error(status, "Error memory copy for antGrid.");


	printGridCluster(objectGrid, objInit, dim);

	int radius = 1;
	int numThreads = 1;
	for(int i = 0; i < 10000000; i++)
	{
		//int clock = 0;
		Timer timer = create_timer();
		start_timer(&timer);
		runDev<<<numAnts, numThreads>>>(clockDev, locksDev, antGridDev, objectGridDev, antArrayDev, objectArrayDev, objectDistDev, dim, numAnts, numObjects, radius);
		check_error( cudaGetLastError(), "Error in kernel.");
		stop_timer(&timer);
		cudaEventSynchronize(timer.stop);

		//status = cudaMemcpy(&clock, clockDev, 1*sizeof(int), cudaMemcpyDeviceToHost);
		//check_error(status, "Error in buffer clockDev reading.");

		cout << "GPU" << ":" << radius << ":" << i << ":" << get_time(&timer) << endl;
		status = cudaMemcpy(objectGrid, objectGridDev, dim * dim * sizeof(int), cudaMemcpyDeviceToHost);
		check_error(status, "Error in buffer objectGrid reading.");

		printGridCluster(objectGrid, objInit, dim);
		destroy_timer(&timer);
	}
	
	cout << colors[0] << "End." << endl;
	return 0;
}
