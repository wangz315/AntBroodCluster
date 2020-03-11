#ifndef _RUN_H
#define _RUN_H

// local headers
#include "global.h"

// cluster the graph
namespace abc
{
	namespace graph
	{
		// original run
		void run();
		// openmp parallel run by critical
		void run_mcc();
		// openmp parallel run by using lock
		void run_mcd();

		void run_mfc();
		void run_mfd();


		// openmp parallel run by critical
		void run_pcc();
		// openmp parallel run by using lock
		void run_pcd();
		
		void run_pfc();
		void run_pfd();
		
		// Random generator
		double rand_gen(long* idum);
		// initialize ant and object grids, and assign ant and objects on the grids
		void init_grids();
		// randomly drop objects on the grid
		void assign_object();
		// assign ants to the objects
		void assign_ant();

		// compute drop off probability 
		double compute_drop(int idi, int x, int y);
		// compute pick up probability 
		double compute_pick(int idi, int x, int y);
		// compute distance between i and j
		double compute_distance(int idi, int idj);
		// compute similarity
		double compute_similarity(int idi, int x, int y);
		// randomly move
		void move(int aid);

		// print grid
		void print_grid();
	}
}

#endif
