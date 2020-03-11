// header files
// local headers
#include "fin.h"


void compute_modularity();

// compute modularity
void compute_modularity()
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
		}
	}

	mod = mod / (2*numEdge);
	cout << "Modularity = " << mod << endl;
}



void fin()
{
	compute_modularity();
}