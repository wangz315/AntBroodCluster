#include "structs.h"

#include <omp.h>
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
#include <string.h>
#include <sstream>
#include <iterator>
#include <iomanip>

#include <sys/time.h>
#include <sys/timeb.h>


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
void move(int antId, int* antGrid, ant_t* antArray, int dim);
double computeProbability(int objectI, int x, int y, int radius, int dim, int numObjects, int* objectGrid, double* objectDist);
int antRun(int antId, int* antGrid, int* objectGrid, ant_t* antArray, object_t* objectArray, double* objectDist, int dim, int numAnts, int numObjects, int radius);

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
	*numAnts = 150;
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



void move(int antId, int* antGrid, ant_t* antArray, int dim)
{

	int factor = 1;
	int randNum = rand() % ((factor*2+1)*(factor*2+1)-1);// %max-1, to avoid move to same location

	//increase 2nd half to avoid same location
	if(randNum >= ((factor*2+1)*(factor*2+1)/2))
	{
		randNum++;
	}

	int x = randNum % (factor*2+1) - factor + antArray[antId].x;
	int y = randNum / (factor*2+1) - factor + antArray[antId].y;

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
	
	#pragma omp critical
	{
		if(antGrid[x*dim+y] == -1)
		{
			antGrid[antArray[antId].x*dim+antArray[antId].y] = -1;
			antArray[antId].x = x;
			antArray[antId].y = y;
			antGrid[x*dim+y] = antId;
		}
	}
}

double computeProbability(int objectI, int x, int y, int radius, int dim, int numObjects, int* objectGrid, double* objectDist)
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

	int area = (pow((radius * 2)+1, 2) - 1);
	double density = total / area;
	density = max(min(density, 1.0), 0.0);
	double temp = exp(-1 * pow(density, 2));
	return  (1-temp) / (1 + temp);
}


int antRun(int antId, int* antGrid, int* objectGrid, ant_t* antArray, object_t* objectArray, double* objectDist, int dim, int numAnts, int numObjects, int radius)
{
	int x = antArray[antId].x;
	int y = antArray[antId].y;
	int pickUpSuccess = 0;
  

	if(objectGrid[x*dim+y] > -1)
	{
		if(antArray[antId].objectId == -1)
		{
			double Ppick = 1 - computeProbability(objectGrid[x*dim+y], x, y, radius, dim, numObjects, objectGrid, objectDist);
			if(Ppick > rand()/(RAND_MAX + 1.))
			{
				antArray[antId].objectId = objectGrid[x*dim+y];
				objectGrid[x*dim+y] = -1;
				pickUpSuccess = 1;

			}
			else
			{
				move(antId, antGrid, antArray, dim);
			}
		}
		else
		{
			move(antId, antGrid, antArray, dim);
		}
	}
	else
	{
		if(antArray[antId].objectId > -1)
		{
			double Pdrop = computeProbability(antArray[antId].objectId, x, y, radius, dim, numObjects, objectGrid, objectDist);
			if(Pdrop > rand()/(RAND_MAX + 1.))
			{
				objectGrid[x*dim+y] = antArray[antId].objectId;
				antArray[antId].objectId = -1;
			}
			else
			{
				move(antId, antGrid, antArray, dim);
			}
		}
		else
		{
			move(antId, antGrid, antArray, dim);
		}
	}
	return pickUpSuccess;
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

	objectInit_t* objInit;
	
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

	printGridCluster(objectGrid, objInit, dim);

	int r = 1;
	for(int i = 0; i < 100000; i++)
	{
		struct timeval s, e;
		gettimeofday (&s, NULL);

		
		for(int j = 0; j < 1000; j++)
		{
				
			#pragma omp parallel for
			for(int a = 0; a < numAnts; a++)
			{

				/*
				int tid = omp_get_thread_num();
				if (tid == 0)
				{
					int nthreads = omp_get_num_threads();
					printf("Number of threads = %d\n", nthreads);
				}
				*/
				
				antRun(a, antGrid, objectGrid, antArray, objectArray, objectDist, dim, numAnts, numObjects, r);

			}
		}

		gettimeofday (&e, NULL);
		long double seconds  = e.tv_sec  - s.tv_sec;
		long double useconds = e.tv_usec - s.tv_usec;

		cout << "r=" << r << ",time=" << (seconds*1000 + useconds/1000)/10 << endl;

		printGridCluster(objectGrid, objInit, dim);
	}

	cout << colors[0] << "End." << endl;
	return 0;
}
