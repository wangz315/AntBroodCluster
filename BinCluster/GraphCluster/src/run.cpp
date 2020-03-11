// header files
// local headers
#include "run.h"


Brood* brood;
int* membership;
int* antArray;
int* objectArray;
int* dropCount;


// functions
// Random generator
double rand_gen(long* idum)
{
	int k;
	double ans;
	k = (*idum)/IQ;
	*idum = IA * (*idum - k * IQ) - IR * k;
	if (*idum < 0 ) * idum += IM;
	ans = AM * (*idum);
	return ans;
}


// initialize ant and object grids, and assign ant and objects on the grids
void init_brood()
{
	brood = new Brood(numObjects);
	antArray = new int[numAnts];
	objectArray = new int[numObjects];
	dropCount = new int[numAnts];

	for(int i = 0; i < numAnts; i++)
	{
		antArray[i] = -1;
		dropCount[i] = 0;
	}

	for(int i = 0; i < numObjects; i++)
	{
		objectArray[i] = i;
	}
}



// compute drop off probability
double compute_drop(int idi, int binId)
{
	double sim = compute_similarity(idi, binId);

	if(sim < k2)
		return 2*sim;
	else
		return 1;
}


// compute pick up probability 
double compute_pick(int idi, int binId)
{
	double sim = compute_similarity(idi, binId);
	return pow(k1 / (sim + k1), 2);
}


// compute distance between i and j
double compute_distance(int idi, int idj)
{
	double diff = 0;
	int cardinaI = 0;
	int cardinaJ = 0;
	double sim;

	for(int i = 0; i < numObjects; i++)
	{
		diff += (double)(adjMatrix[idi*numObjects + i] ^ adjMatrix[idj*numObjects + i]);

		if(adjMatrix[idi*numObjects + i])
		{
			cardinaI++;
		}

		if(adjMatrix[idj*numObjects + i])
		{
			cardinaJ++;
		}
	}

	if(cardinaI + cardinaJ == 0)
	{
		sim = 0;
	}
	else
	{
		sim = diff / (double)(cardinaI + cardinaJ);
	}

	return sim;
}



// compute similarity
double compute_similarity(int idi, int binId)
{
	int idj;
	double sim = 0.0;
	double dist = 0.0;

	Bin* bin = brood->get_bin(binId);

	Node* curr = bin->head;

	for(int i = 0; i < bin->size; i++)
	{
		idj = curr->objectId;

		if(idi != idj)
		{
			dist = compute_distance(idi, idj);
			sim += (1 - dist/alpha);
		}
		
		curr = curr->next;
	}
	
	sim = sim/bin->size;

	if(sim < 0)
	{
		sim = 0;
	}

	return sim;
}



void bin_refill(int idi, int idj)
{
	Bin* bin1 = brood->get_bin(idi);
	Bin* bin2 = brood->get_bin(idj);
	Bin* tmpBin = new Bin();
	int count = 0;

	Node* node = bin1->head;

	for(int i = 0; i < bin1->size; i++)
	{
		int id = node->objectId;

		double p = compute_drop(id, idj);
		double r = ((double) rand() / (RAND_MAX));

		if(p > r)
		{	
			count++;
		}

		tmpBin->insert(id);
		node = node->next;
	}

	node = tmpBin->head;

	if(count >= bin1->size * 0.8)
	{
		for(int i = 0; i < tmpBin->size; i++)
		{
			
			int id = node->objectId;

			bin1->remove(id);
			if(bin1->size == 0)
			{
				brood->delete_bin(bin1->id);
			}
			bin2->insert(id);

			objectArray[id] = bin2->id;

			node = node->next;
		}
	}

	delete tmpBin;
}


// assign membership
void assign_member()
{
	membership = new int[numObjects];

	Bin* bin = brood->head;
	for(int i = 0; i < brood->size; i++)
	{
		Node* node = bin->head;
		for(int j = 0; j < bin->size; j++)
		{
			membership[node->objectId] = i;
			node = node->next;
		}

		bin = bin->next;
	}	
}



void run()
{
	srand(time(NULL));
	init_brood();

	for(int i = 0; i < numIter; i++)
	{
		for(int a = 0; a < numAnts; a++)
		{
			if(antArray[a] == -1)
			{
				// try pick up
				int id;
				do
				{
					id = rand()%numObjects;
				}
				while(objectArray[id] == -1);

				double p = compute_pick(id, objectArray[id]);
				double r = ((double) rand() / (RAND_MAX));

				if(p > r)
				{
					
					brood->get_bin(objectArray[id])->remove(id);

					if(brood->get_bin(objectArray[id])->size == 0)
					{
						brood->delete_bin(objectArray[id]);
					}


					antArray[a] = id;
					objectArray[id] = -1;

				}

			}
			else
			{
				// try drop off
				int id = antArray[a];
				Bin* bin = brood->rand_bin();

				double p = compute_drop(id, bin->id);
				double r = ((double) rand() / (RAND_MAX));

				if(p > r)
				{
					bin->insert(id);
					antArray[a] = -1;
					objectArray[id] = bin->id;
					dropCount[a] = 0;
				}
				// else
				// {
				// 	dropCount[a]++;

				// 	if(dropCount[a] > brood->size)
				// 	{
				// 		int binId = brood->create_bin();
				// 		Bin* newBin = brood->get_bin(binId);
				// 		newBin->insert(id);
				// 		antArray[a] = -1;
				// 		objectArray[id] = binId;
				// 		dropCount[a] = 0;
				// 	}
				// }
			}
		}
	}

	for(int a = 0; a < numAnts; a++)
	{
		while(antArray[a] != -1)
		{
			int id = antArray[a];
			Bin* bin = brood->rand_bin();

			double p = compute_drop(id, bin->id);
			double r = ((double) rand() / (RAND_MAX));

			if(p > r)
			{
				bin->insert(id);
				antArray[a] = -1;
				objectArray[id] = bin->id;
			}
		}
	}

	brood->print();
	cout << "=========" << endl;

	for(int i = 0; i < interval; i++)
	{
		for(int a = 0; a < numAnts; a++)
		{
			int idi = brood->rand_bin()->id;
			int idj = brood->rand_bin()->id;

			while(idi == idj)
			{
				idj = brood->rand_bin()->id;
			}
			

			bin_refill(idi, idj);
		}
	}

	assign_member();

	brood->print();
}
