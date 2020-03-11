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

double* features;
int featureLength;


// algorithm parameters
int s; // radius
double alpha;
double k1;
double k2;
int k;



//========================================================================================//
// prototypes
void read_parameter();
void read_gr();
void set_parameter();
void convert_objInfo(string str, int index);
// read data format of csv
void read_csv();



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




void set_parameter()
{
	ifstream fin; // input file stream
	string str; // temp string for read
	numObjects = 0;
	featureLength = 0;

	fin.open(fileName);


	getline(fin, str);
	numObjects++;

	// count the feature length by number of separators - ','
	for(unsigned int i = 0; i < str.length(); i++)
	{
		if(str.c_str()[i] == ',')
		{
			featureLength++;
		}
	}

	// count the remainning lines
	while(getline(fin, str))
	{
		if(isdigit(str.c_str()[0]))
		{
			numObjects++;
		}
	}


	fin.close();
}


// convert input line into objects (obj_info)
void convert_objInfo(string str, int index)
{
	string token;
	istringstream iss(str);

	getline(iss, token, ',');
	token = token.substr(token.find_first_of("0123456789")); // trim, for the first line in the file, there are some dirty chars.
	features[index*numObjects+featureLength] = atoi(token.c_str());
			

	for(int i = 0; i < featureLength; i++)
	{
		getline(iss, token, ',');
		features[index*numObjects+i] = atof(token.c_str());
	}
}



// read data format of csv
void read_csv()
{
	ifstream fin; // input file stream
	string str; // temp string for read

	set_parameter(); // allocate memory, and so on
			
	fin.open(fileName);

	// allocate objects and their features, the last index is cluster id
	features = new double[numObjects*(featureLength+1)];

	// read object information from file
	for(int i = 0; i < numObjects; i++)
	{
		getline(fin, str);
		convert_objInfo(str, i);
	}

	fin.close();
}


// read graph to objects
void init(int argc, char **argv)
{
	argArray = argv;
	argLength = argc;

	read_parameter();
	read_csv();

}