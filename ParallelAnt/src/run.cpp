// header files
// local headers
#include "run.h"

namespace abc
{
	namespace graph
	{
		//========================================================================================//
		// prototypes
		

		//========================================================================================//
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
		void init_grids()
		{
			antArray = new ant_t[numAnts];
			objectArray = new object_t[numObjects];
			antGrid = new int[dim * dim];
			objectGrid = new int[dim * dim];

			assign_object();
			assign_ant();
		}


		// randomly drop objects on the grid
		void assign_object()
		{
			int x, y, posTaken;

			// init grid
			for(int i = 0; i < dim * dim; i++)
			{
				objectGrid[i] = -1;
			}

			// drop object
			for(int i = 0; i < numObjects; i++)
			{
				posTaken = 0;
				while(posTaken == 0)
				{
					x = rand() % dim;
					y = rand() % dim;
					if(objectGrid[x * dim + y] < 0)
					{
						posTaken = 1;
						objectGrid[x * dim + y] = i;
						objectArray[i].x = x;
						objectArray[i].y = y;
					}
				}
			}
		}


		// assign ants to the objects
		void assign_ant()
		{
			for(int i = 0; i < dim * dim; i++)
			{
				antGrid[i] = -1;
			}

			for(int i = 0; i < numAnts; i++)
			{
				antArray[i].x = objectArray[i].x;
				antArray[i].y = objectArray[i].y;
				antArray[i].objectId = -1;
				antArray[i].seed = rand();

				antGrid[antArray[i].x * dim + antArray[i].y] = i;
			}
		}



		// compute drop off probability
		double compute_drop(int idi, int x, int y)
		{
			double sim = compute_similarity(idi, x, y);

			if(sim < k2)
				return 2*sim;
			else
				return 1;
		}


		// compute pick up probability 
		double compute_pick(int idi, int x, int y)
		{
			double sim = compute_similarity(idi, x, y);
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
		double compute_similarity(int idi, int x, int y)
		{
			int idj;
			int xi, yj;
			int xs = x - s/2;
			int ys = y - s/2;
			double sim = 0.0;
			double dist = 0.0;

			for(int i = 0; i < s; i++)
			{
				xi = xs + i;
				if(xi > -1 && xi < dim)
				{
					for(int j = 0; j < s; j++)
					{
						yj = ys + j;

						if(yj > -1 && yj < dim)
						{
							idj = objectGrid[xi * dim + yj];
							if(idj > -1 && idi != idj)
							{
								dist = compute_distance(idi, idj);
								sim += (1 - dist/alpha);
							}
						}  
					}
				}
			}


			sim = sim/(s * s - 1.0);

			if(sim < 0)
			{
				sim = 0;
			}
			

			return sim;
		}


		// randomly move
		void move(int aid)
		{

			int x = antArray[aid].x;
			int y = antArray[aid].y;
			int randNum = rand_gen(&antArray[aid].seed) * 4;

			if(randNum == 0)
			{
				x = antArray[aid].x + 1;
				
				if(x >= dim)
				{
					move(aid);
					return;
				}
			}
			else if(randNum == 1)
			{
				y = antArray[aid].y + 1;
				
				if(y >= dim)
				{
					move(aid);
					return;
				}
			}
			else if(randNum == 2)
			{
				x = antArray[aid].x - 1;

				if(x < 0)
				{
					move(aid);
					return;
				}
			}
			else
			{
				y = antArray[aid].y - 1;

				if(y < 0)
				{
					move(aid);
					return;
				}
			}

			//#pragma omp critical
			{
				if(antGrid[x*dim+y] == -1)
				{
					antGrid[antArray[aid].x * dim + antArray[aid].y] = -1;
					antArray[aid].x = x;
					antArray[aid].y = y;
					antGrid[x * dim + y] = aid;
				}
			}

		}


		// start clustering
		void run()
		{
			//srand(1);
			srand(time(NULL));

			init_grids();


			start_time = omp_get_wtime();
			{

				for(int i = 0; i < numIter/interval; i++)
				{

					for(int j = 0; j < interval; j++)// print interval
					{
						
						for(int a = 0; a < numAnts; a++)
						{
							
							int x = antArray[a].x;
							int y = antArray[a].y;

							int id = objectGrid[x * dim + y];

							// if grid site is not empty, and ant is unladen, try pick up
							if(id > -1 && antArray[a].objectId == -1)
							{
								double p = compute_pick(id, x, y);
								double r = rand_gen(&antArray[a].seed);

								if(p > r)
								{
									//#pragma omp critical
									if(objectGrid[x * dim + y] > -1)
									{
										antArray[a].objectId = id;
										objectGrid[x * dim + y] = -1;
									}
								}
							}
							// if grid site is empty, and ant is laden, try drop off
							else if(id == -1 && antArray[a].objectId > -1)
							{
								id = antArray[a].objectId;
								double p = compute_drop(id, x, y);
								double r = rand_gen(&antArray[a].seed);
								if(p > r)
								{
									//#pragma omp critical
									if(objectGrid[x * dim + y] == -1)
									{
										objectGrid[x * dim + y] = antArray[a].objectId;
										antArray[a].objectId = -1;
									}
								}
							}

							// random move
							move(a);
						}
					}
				}

				// clean up
				//#pragma omp parallel for
				// for(int a = 0; a < numAnts; a++)
				// {
				// 	int count = 0;
				// 	while(antArray[a].objectId > -1)
				// 	{
				// 		int x = antArray[a].x;
				// 		int y = antArray[a].y;

				// 		int id = objectGrid[x * dim + y];

				// 		if(id == -1)
				// 		{
				// 			id = antArray[a].objectId;
				// 			double p = compute_drop(id, x, y);
				// 			double r = rand_gen(&antArray[a].seed);

				// 			if(p > r || count > dim*dim)
				// 			{

				// 				//#pragma omp critical
				// 				if(objectGrid[x * dim + y] == -1)
				// 				{
				// 					objectGrid[x * dim + y] = antArray[a].objectId;
				// 					antArray[a].objectId = -1;
				// 				}
				// 			}

				// 			count++;
				// 		}

				// 		// random move
				// 		move(a);
				// 	}
				// }
			}
			stop_time = omp_get_wtime();
			
			//print_grid();
		}


		// print grid with colors
		void print_grid()
		{
			string colors[11] = {"\033[0m", "\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[35m", "\033[36m", "\033[37m", "\033[38m", "\033[39m", "\033[40m"};

			cout << colors[0];
			for(int i = 0; i < dim * 3; i++)
			{
				cout << "=";
			}
			cout << endl;

			for(int i = 0; i < dim; i++)
			{
				for(int j = 0; j < dim; j++)
				{
					if(objectGrid[i * dim + j] > -1)
					{
						cout << setw(3) << objectGrid[i * dim + j]+1;
					}
					else
					{
						cout << setw(3) << "  ";
					}
				}
				cout << endl;
			}
			
			cout << colors[0];
			for(int i = 0; i < dim * 3; i++)
			{
				cout << "=";
			}
			cout << endl;
		}

	}
}