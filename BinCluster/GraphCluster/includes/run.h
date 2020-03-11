#ifndef _RUN_H
#define _RUN_H

// local headers
#include "globals.h"
#include "brood.h"

// original run
void run();
		
// Random generator
double rand_gen(long* idum);
// initialize ant and object grids, and assign ant and objects on the grids
void init_brood();

// compute drop off probability 
double compute_drop(int idi, int binId);
// compute pick up probability 
double compute_pick(int idi, int binId);
// compute distance between i and j
double compute_distance(int idi, int idj);
// compute similarity
double compute_similarity(int idi, int binId);
// assign membership
void assign_member();


#endif
