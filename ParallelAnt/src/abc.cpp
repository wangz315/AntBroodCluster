// header files
// local headers
#include "abc.h"

namespace abc
{
	namespace graph
	{
		// cluster the graph, original algorithm
		void cluster()
		{
			init();
			run();
			fin();
		}


		// openmp parallel algorithm
		void cluster_mcc()
		{
			init();
			run_mcc();
			fin();
		}


		// openmp parallel algorithm by lock
		void cluster_mcd()
		{
			init();
			run_mcd();
			fin();
		}

		void cluster_mfc()
		{
			init();
			run_mfc();
			fin();
		}


		void cluster_mfd()
		{
			init();
			run_mfd();
			fin();
		}


		// openmp parallel algorithm
		void cluster_pcc()
		{
			init();
			run_pcc();
			fin();
		}


		// openmp parallel algorithm by lock
		void cluster_pcd()
		{
			init();
			run_pcd();
			fin();
		}

		void cluster_pfc()
		{
			init();
			run_pfc();
			fin();
		}


		void cluster_pfd()
		{
			init();
			run_pfd();
			fin();
		}
	}
}