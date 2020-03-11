// header files
// local headers
#include "init.h"


//========================================================================================//
// global variables
int argLength;
char** argArray;


// parameters
char* fileName; // input file name
int dim; // grid size
int numAnts;
int numObjects;
int numEdge;
int numIter; // number of iterations
int interval; // intermid interval
int* adjMatrix;


// algorithm parameters
int s; // radius
double alpha;
double k1;
double k2;
double p;
int k;



//========================================================================================//
// prototypes
void read_parameter();
void read_gr();




//========================================================================================//
// functions
void read_parameter()
{
	int index = 1;

	s = 3;
	alpha = 1;
	k1 = 0.3;
	k2 = 0.1;


	while(index < argLength)
	{
		if(!strcmp("-f", argArray[index]))
		{
			fileName = argArray[index+1];
		}
		else if(!strcmp("-d", argArray[index]))
		{
			dim = atoi(argArray[index+1]);
		}
		else if(!strcmp("-a", argArray[index]))
		{
			numAnts = atoi(argArray[index+1]);
		}
		else if(!strcmp("-n", argArray[index]))
		{
			numObjects = atoi(argArray[index+1]);
		}
		else if(!strcmp("-i", argArray[index]))
		{
			numIter = atoi(argArray[index+1]);
		}
		else if(!strcmp("-m", argArray[index]))
		{
			interval = atoi(argArray[index+1]);
		}
		else if(!strcmp("-s", argArray[index]))
		{
			s = atoi(argArray[index+1]);
		}
		else if(!strcmp("-l", argArray[index]))
		{
			alpha = atof(argArray[index+1]);
		}
		else if(!strcmp("-1", argArray[index]))
		{
			k1 = atof(argArray[index+1]);
		}
		else if(!strcmp("-2", argArray[index]))
		{
			k2 = atof(argArray[index+1]);
		}
		else if(!strcmp("-k", argArray[index]))
		{
			k = atoi(argArray[index+1]);
		}
		else if(!strcmp("-p", argArray[index]))
		{
			p = atof(argArray[index+1]);
		}


		index++;
	}
}



// read data format of gr
void read_gr()
{
	ifstream fin; // input file stream
	string str; // temp string for read
	char var1[2], var2[2];
	int s, t;


	fin.open(fileName);

	getline(fin, str); // read comments
	getline(fin, str); // read parameters

	sscanf(str.c_str(), "%s %s %d %d", var1, var2, &numObjects, &numEdge);

	adjMatrix = new int[numObjects * numObjects];

	while(getline(fin, str))
	{
		
		sscanf(str.c_str(), "a %d %d", &s, &t);

		if(!strcmp("graphs/15nodeSample.gr", fileName))
		{
			s--;
			t--;
		}

		adjMatrix[s*numObjects+t] = 1;
	}

	if(!strcmp("graphs/15nodeSample.gr", fileName))
		numEdge /= 2;
	
	fin.close();
}


// read graph to objects
void init(int argc, char **argv)
{
	argArray = argv;
	argLength = argc;

	read_parameter();
	read_gr();

}