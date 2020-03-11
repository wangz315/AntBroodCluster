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


// compute drop off probability by KDE
double compute_drop(int idi, int binId)
{
	int idj;
	double sim = 0.0;
	double total = 0.0;


	Bin* bin = brood->get_bin(binId);

	Node* curr = bin->head;

	for(int i = 0; i < bin->size; i++)
	{
		idj = curr->objectId;

		if(idi != idj)
		{
			sim = compute_distance(idi, idj);
			total += exp(-1 * sim);
		}
		
		curr = curr->next;
	}


	int area = bin->size;
	double density = total / area;
	density = max(min(density, 1.0), 0.0);
	double p = exp(-1 * pow(density, 2));

	return  (1 - p) / (1 + p);
}


// compute pick up probability 
double compute_pick(int idi, int binId)
{
	return 1 - compute_drop(idi, binId);
}


// compute distance between i and j
double compute_distance(int idi, int idj)
{
	double sum = 0.0;

	for(int i = 0; i < featureLength; i++)
	{
		sum = sum + pow((features[idi*numObjects+i] - features[idj*numObjects+i]), 2.0);
	}

	return sqrt(sum);
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

	if(count >= bin1->size * k1)
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


void bin_merge(int idi, int idj)
{
	Bin* bin1 = brood->get_bin(idi);
	Bin* bin2 = brood->get_bin(idj);

	double sim = 0;
	double total = 0.0;

	Node* node1 = bin1->head;
	Node* node2;

	for(int i = 0; i < bin1->size; i++)
	{
		int obji = node1->objectId;
		node2 = bin2->head;
		for(int j = 0; j < bin2->size; j++)
		{
			int objj = node2->objectId;

			sim += compute_distance(obji, objj);
			total += exp(-1 * sim);

			node2 = node2->next;
		}
		node1 = node1->next;
	}


	int area = 1;
	double density = total / area;
	density = max(min(density, 1.0), 0.0);
	double p = exp(-1 * pow(density, 2));
	p = (1 - p) / (1 + p);

	//cout << idi << " " << idj << " " << p << endl;

	double r = ((double) rand() / (RAND_MAX));

	if(p > r)
	{
		int binId = brood->create_bin();
		Bin* newBin = brood->get_bin(binId);
		node1 = bin1->head;
		

		for(int i = 0; i < bin1->size; i++)
		{
			newBin->insert(node1->objectId);
			objectArray[node1->objectId] = binId;
			node1 = node1->next;
		}

		node1 = bin2->head;
		for(int i = 0; i < bin2->size; i++)
		{
			newBin->insert(node1->objectId);
			objectArray[node1->objectId] = binId;
			node1 = node1->next;
		}

		brood->delete_bin(idi);
		brood->delete_bin(idj);

		delete bin1;
		delete bin2;
	}
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

	cout << "===============" << endl;

	assign_member();
}
