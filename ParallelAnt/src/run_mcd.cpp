// header files
// local headers
#include "run.h"

namespace abc
{
	namespace graph
	{
		//========================================================================================//
		// global variables

		// prototypes
		// init lock parameters
		void init_mcd();
		// randomly move
		void move_mcd(int aid);
		

		//========================================================================================//
		// functions
		void init_mcd()
		{
			lockGrid = (omp_lock_t*) malloc(sizeof(omp_lock_t) * dim * dim);

			for(int i = 0; i < dim * dim; i++)
			{
				omp_init_lock(&lockGrid[i]);
			}
		}


		// randomly move
		void move_mcd(int aid)
		{

			int x = antArray[aid].x;
			int y = antArray[aid].y;
			int randNum = rand_gen(&antArray[aid].seed) * 4;

			if(randNum == 0)
			{
				x = antArray[aid].x + 1;
				
				if(x >= dim)
				{
					move_mcd(aid);
					return;
				}
			}
			else if(randNum == 1)
			{
				y = antArray[aid].y + 1;
				
				if(y >= dim)
				{
					move_mcd(aid);
					return;
				}
			}
			else if(randNum == 2)
			{
				x = antArray[aid].x - 1;

				if(x < 0)
				{
					move_mcd(aid);
					return;
				}
			}
			else
			{
				y = antArray[aid].y - 1;

				if(y < 0)
				{
					move_mcd(aid);
					return;
				}
			}


			{
				omp_set_lock(&lockGrid[x * dim + y]);
				if(antGrid[x*dim+y] == -1)
				{
					//omp_set_lock(&lockGrid[antArray[aid].x * dim + antArray[aid].y]);
					antGrid[antArray[aid].x * dim + antArray[aid].y] = -1;
					//omp_unset_lock(&lockGrid[antArray[aid].x * dim + antArray[aid].y]);
					antArray[aid].x = x;
					antArray[aid].y = y;
					antGrid[x * dim + y] = aid;
				}
				omp_unset_lock(&lockGrid[x * dim + y]);
			}

		}


		// start clustering
		void run_mcd()
		{
			//srand(1);
			srand(time(NULL));

			//double start_time, stop_time;
			omp_set_num_threads(numAnts);
			init_grids();
			init_mcd();

			start_time = omp_get_wtime();
			#pragma omp parallel
			{
				int a = omp_get_thread_num();

				for(int i = 0; i < numIter/interval; i++)
				{
					//start_time = omp_get_wtime();
					for(int j = 0; j < interval; j++)// print interval
					{
						
						//#pragma omp parallel for
						//for(int a = 0; a < numAnts; a++)
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
									//omp_set_lock(&lockGrid[x * dim + y]);
									if(objectGrid[x * dim + y] > -1)
									{
										antArray[a].objectId = id;
										objectGrid[x * dim + y] = -1;
									}
									//omp_unset_lock(&lockGrid[x * dim + y]);
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
									//omp_set_lock(&lockGrid[x * dim + y]);
									if(objectGrid[x * dim + y] == -1)
									{
										objectGrid[x * dim + y] = antArray[a].objectId;
										antArray[a].objectId = -1;
									}
									//omp_unset_lock(&lockGrid[x * dim + y]);
								}
							}

							// random move
							move_mcd(a);
						}
					}
				}

				// clean up
				//#pragma omp parallel for
				//for(int a = 0; a < numAnts; a++)
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

				// 				omp_set_lock(&lockGrid[x * dim + y]);
				// 				if(objectGrid[x * dim + y] == -1)
				// 				{
				// 					objectGrid[x * dim + y] = antArray[a].objectId;
				// 					antArray[a].objectId = -1;
				// 				}
				// 				omp_unset_lock(&lockGrid[x * dim + y]);
				// 			}

				// 			count++;
				// 		}

				// 		// random move
				// 		move_mcd(a);
				// 	}
				// }
			}
			stop_time = omp_get_wtime();

			for(int i = 0; i < dim * dim; i++)
			{
				omp_destroy_lock(&lockGrid[i]);
			}

			//print_grid();
		}

	}
}