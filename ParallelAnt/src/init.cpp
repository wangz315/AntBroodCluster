// header files
// local headers
#include "init.h"


//========================================================================================//
// global variables
// parameters
string fileName; // input file name
int dim; // grid size
int numAnts;
int numObjects;
int numIter; // number of iterations
int interval; // intermid interval

// common
int* objectGrid;
int* antGrid;
object_t* objectArray; // location of objects
ant_t* antArray; // location, objId and seed of ants

// data file
objectInfo_t* objInfo; // object information from plain
int featureLength; // used to compute the distances

// graph file
int* adjMatrix;
int numEdge;

// algorithm parameters
int s; // radius
double* dist; // distance matrix
double alpha;
double k1;
double k2;

omp_lock_t* lockGrid;

// information retrieval
int* membership; // cluster membership
int* kcentroids;
int k;

double modularity;
double interDist;
double intraDist;

double start_time;
double stop_time;

namespace abc
{
	namespace graph
	{
		//========================================================================================//
		// prototypes
		// init the parameters to default values
		void init_parameter();
		// read config file to set the parameters
		void read_config();
		// read data format of gr
		void read_gr();
		// read data format of gr
		void read_graph();
		// read data from file
		void read_file();

		//========================================================================================//
		// functions
		// init the parameters to default values
		void init_parameter()
		{
			fileName = "_UNDEFINED_"; // input file name

			interval = 2000;
			dim = UNDEFINED; // grid size
			numAnts = UNDEFINED;
			numObjects = UNDEFINED;
			numEdge = UNDEFINED;
			numIter = UNDEFINED;

			objectGrid = NULL;
			antGrid = NULL;
			objectArray = NULL;
			antArray = NULL;

			alpha = 1;
			k1 = 0.3;
			k2 = 0.1;
			s = 3;
			k = 3;

			adjMatrix = NULL;
		}


		// read config file to set the parameters
		void read_config()
		{
			string config = "config"; // config file name
			ifstream fin; // input file stream
			string str;
			char key[20], value[20];

			fin.open(config.c_str());

			while(getline(fin, str))
			{
				sscanf(str.c_str(), "%s = %s", key, value);

				if(!strcmp("file", key))
				{
					fileName = string(value);
				}
				else if(!strcmp("interval", key))
				{
					interval = atoi(value);
				}
				else if(!strcmp("iter", key))
				{
					if(atoi(value) > 0)
					{
						numIter = atoi(value);
					}
				}
				else if(!strcmp("ants", key))
				{
					if(atoi(value) > 0)
					{
						numAnts = atoi(value);
					}
				}
				else if(!strcmp("dim", key))
				{
					if(atoi(value) > 0)
					{
						dim = atoi(value);
					}
				}
				else if(!strcmp("alpha", key))
				{
					if(atof(value) > 0.0)
					{
						alpha = atof(value);
					}
				}
				else if(!strcmp("k1", key))
				{
					if(atof(value) > 0.0)
					{
						k1 = atof(value);
					}
				}
				else if(!strcmp("k2", key))
				{
					if(atof(value) > 0.0)
					{
						k2 = atof(value);
					}
				}
				else if(!strcmp("s", key))
				{
					if(atoi(value) > 0)
					{
						s = atoi(value);
					}
				}
				else if(!strcmp("k", key))
				{
					if(atoi(value) > 0)
					{
						k = atoi(value);
					}
				}
			}

			fin.close();
		}


		// read data format of gr
		void read_gr()
		{
			ifstream fin; // input file stream
			string str; // temp string for read
			char var1[2], var2[2];
			int s, t;

			
			
			fin.open(fileName.c_str());



			getline(fin, str); // read comments
			getline(fin, str); // read parameters

			sscanf(str.c_str(), "%s %s %d %d", var1, var2, &numObjects, &numEdge);

			
			adjMatrix = new int[numObjects * numObjects];
			

			while(getline(fin, str))
			{
				if(!strcmp("graphs/sample.gr", fileName.c_str()))
				{
					int w;
					sscanf(str.c_str(), "a %d %d %d", &s, &t, &w);
				}
				else
				{
					sscanf(str.c_str(), "a %d %d", &s, &t);
				}
				

				if(!strcmp("graphs/15nodeSample.gr", fileName.c_str()) || !strcmp("graphs/sample.gr", fileName.c_str()))
				{
					s--;
					t--;
				}

				adjMatrix[s*numObjects+t] = 1;
			}

			if(!strcmp("graphs/15nodeSample.gr", fileName.c_str()))
				numEdge /= 2;

			if(numAnts == UNDEFINED)
			{
				numAnts = numObjects/10 + 1;
			}
			
			if(dim == UNDEFINED)
			{
				dim = sqrt(numObjects*10) + 1;
			}

			if(numIter == UNDEFINED)
			{
				numIter = 2000 * numObjects;
			}
			
			fin.close();
		}


		// read data format of graph
		void read_graph()
		{
			ifstream fin; // input file stream
			string str; // temp string for read
			char* token;
			char buffer[1024];
			int fmt;
			int s = 0, t;

			
			
			fin.open(fileName.c_str());

			getline(fin, str); // read parameters

			sscanf(str.c_str(), "%d %d %d", &numObjects, &numEdge, &fmt);

			
			adjMatrix = new int[numObjects * numObjects];
			

			while(getline(fin, str))
			{
				if(str.length() > 1)
				{
					strcpy(buffer, str.c_str());
					token = strtok(buffer, " ");

					while(token != NULL)
					{
						t = atoi(token) - 1;

						adjMatrix[s*numObjects+t-1] = 1;

						if(fmt)
						{
							token = strtok(NULL, " ");// weight
						}

						token = strtok(NULL, " ");
					}
					
				}
				s++;
			}
			

			if(numAnts == UNDEFINED)
			{
				numAnts = numObjects/10 + 1;
			}
			
			if(dim == UNDEFINED)
			{
				dim = sqrt(numObjects*10) + 1;
			}

			if(numIter == UNDEFINED)
			{
				numIter = 2000 * numObjects;
			}
			
			fin.close();

			//delete [] buffer;
		}


		// read data from file
		void read_file()
		{

			if(fileName.substr(fileName.find_last_of(".") + 1) == "gr")
			{
				read_gr();
			}
			else if(fileName.substr(fileName.find_last_of(".") + 1) == "graph")
			{
				read_graph();
			}
			
		}


		// read graph to objects
		void init()
		{
			// init the parameters to default values
			init_parameter();
			// read config file to set the parameters
			read_config();
			// read data from file
			read_file();
		}
	}
}