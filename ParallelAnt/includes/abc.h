#ifndef _ANT_BROOD_CLUSTER_H
#define _ANT_BROOD_CLUSTER_H

// local headers
#include "init.h"
#include "run.h"
#include "fin.h"

namespace abc
{
	namespace graph
	{
		// cluster the graph, original algorithm
		void cluster();
		// openmp parallel algorithm
		void cluster_mcc();
		// openmp parallel algorithm by lock
		void cluster_mcd();

		void cluster_mfc();
		void cluster_mfd();

		// openmp parallel algorithm
		void cluster_pcc();
		// openmp parallel algorithm by lock
		void cluster_pcd();

		void cluster_pfc();
		void cluster_pfd();
	}
}


#endif