// header files
// local headers
#include "fin.h"


namespace abc
{
	namespace graph
	{
		//========================================================================================//
		// prototypes
		// cluster information retrieval by k means
		void info_retrieval();
		// compute modularity
		double compute_mod();
		// print results
		void print_result();

		//========================================================================================//
		// functions
		// cluster information retrieval by k means
		void info_retrieval()
		{
			double maxError = 0.1;
			double centroids[k*2];
			double tempCentroids[k*2];
			int count[k*3];//index0: number of elements in the cluster, index1: sum of x coordinates, index2: sum of y coordinates
			int clusters[k*numObjects];
			int objJ;
			double error = INT_MAX;
			membership = new int[numObjects];

			int tempMem, tempCount;

			// update the locations of objects
			for(int i = 0; i < dim; i++)
			{
				for(int j = 0; j < dim; j++)
				{
					int id = objectGrid[i*dim+j];

					if(id > -1)
					{
						objectArray[id].x = i;
						objectArray[id].y = j;
					}
				}
			}

			// random assign centroids
			for(int i = 0; i < k; i++)
			{
				centroids[i*2] = rand()%dim;
				centroids[i*2+1] = rand()%dim;
			}


			// while centroids do not change anymore
			while(error > maxError)
			{
				
				for(int i = 0; i < k; i++)
				{
					count[i*3] = 0;
					count[i*3+1] = 0;
					count[i*3+2] = 0;
				}


				for(int i = 0; i < numObjects; i++)
				{
					double minDist = INT_MAX;
					double distance;

					// find the closest cent
					for(int j = 0; j < k; j++)
					{
						distance = sqrt(pow((objectArray[i].x - centroids[j*2]), 2) + pow((objectArray[i].y - centroids[j*2+1]), 2));

						if(distance < minDist)
						{
							membership[i] = j;
							minDist = distance;
						}
					}
					
					tempMem = membership[i];
					tempCount = count[tempMem*3];
					count[tempMem*3]++;
					count[tempMem*3+1]+=objectArray[i].x;
					count[tempMem*3+2]+=objectArray[i].y;
					clusters[tempMem*numObjects+tempCount] = i;


				}


				for(int i = 0; i < k*2; i++)
				{
					tempCentroids[i] = centroids[i];
				}

				error = 0;

				for(int i = 0; i < k; i++)
				{
					if(count[i*3] > 0)
					{
						centroids[i*2]=count[i*3+1]*1.0/count[i*3];
						centroids[i*2+1]=count[i*3+2]*1.0/count[i*3];
					}
					else
					{
						//If no one is in this cluster, make the centroid at 0
						centroids[i*2]=0;
						centroids[i*2+1]=0;
					}

					error+= abs(centroids[i*2]-tempCentroids[i*2]);
					error+= abs(centroids[i*2+1]-tempCentroids[i*2+1]);
				}

				error=error/(2*k);
			}

			kcentroids = new int[k * 2];

			for(int i = 0; i < k; i++)
			{
				kcentroids[i*2] = centroids[i*2];
				kcentroids[i*2+1] = centroids[i*2+1];
			}


			modularity = compute_mod();

			interDist = 0;
			intraDist = 0;
			double tmpDist;

			for(int i = 0; i < k; i++)
			{
				for(int j = 0; j < k; j++)
				{
					if(i != j)
					{
						interDist += (pow((centroids[i*2]-centroids[j*2]),2) + pow((centroids[i*2+1]-centroids[j*2+1]),2));
					}
				}
			}

			interDist /= 2;

			for(int i = 0; i < k; i++)
			{
				tmpDist= 0;

				for(int j = 0; j < count[i*3]; j++)
				{
					objJ = clusters[i*numObjects+j];
					tmpDist += (pow((objectArray[objJ].x - centroids[i*2]), 2) + pow((objectArray[objJ].y - centroids[i*2+1]), 2));
				}

				tmpDist /= count[i*3];
				intraDist += tmpDist;
			}

		}


		// compute modularity
		double compute_mod()
		{
			double mod;
			int cardinalityI, cardinalityJ;

			mod = 0;

			for(int i = 0; i < numObjects; i++)
			{
				for(int j = 0; j < numObjects; j++)
				{
					if(membership[i] == membership[j])
					{
						cardinalityI = 0;
						cardinalityJ = 0;

						for(int index = 0; index < numObjects; index++)
						{
							if(adjMatrix[i*numObjects+index] == 1)
							{
								cardinalityI++;
							}

							if(adjMatrix[j*numObjects+index] == 1)
							{
								cardinalityJ++;
							}
						}

						mod = mod + (adjMatrix[i*numObjects+j] - (double)(cardinalityI*cardinalityJ)/(double)(2*numEdge));
					}
					else
					{
						mod += 0;
					}
				}
			}

			mod = mod / (2*numEdge);
			return mod;
		}


		// print results
		void print_result()
		{
			//cout << "Data Size  = " << "\033[33m" << numObjects << "\033[0m" << endl;
			//cout << "Modularity = " << "\033[31m" << modularity << "\033[0m" << endl;
			cout << "Time cost  = " << "\033[32m" << stop_time - start_time << "\033[0m" << endl;
		}

		void fin()
		{
			info_retrieval();
			print_result();
		}
	}
}