// header files
// local headers
#include "abc.h"

// parameters reader
void reader(int argc, char **argv)
{
	if(argc == 1)
	{
		cout << "========== ABC ==========" << endl;
		abc::graph::cluster();
	}
	else if(!strcmp(argv[1],"mcc"))
	{
		cout << "=========== MCC ===========" << endl;
		abc::graph::cluster_mcc();
	}
	else if(!strcmp(argv[1],"mcd"))
	{
		cout << "=========== MCD ===========" << endl;
		abc::graph::cluster_mcd();
	}
	else if(!strcmp(argv[1],"mfc"))
	{
		cout << "=========== MFC ===========" << endl;
		abc::graph::cluster_mfc();
	}
	else if(!strcmp(argv[1],"mfd"))
	{
		cout << "=========== MFD ===========" << endl;
		abc::graph::cluster_mfd();
	}
	else if(!strcmp(argv[1],"pcc"))
	{
		cout << "=========== PCC ===========" << endl;
		abc::graph::cluster_pcc();
	}
	else if(!strcmp(argv[1],"pcd"))
	{
		cout << "=========== PCD ===========" << endl;
		abc::graph::cluster_pcd();
	}
	else if(!strcmp(argv[1],"pfc"))
	{
		cout << "=========== PFC ===========" << endl;
		abc::graph::cluster_pfc();
	}
	else if(!strcmp(argv[1],"pfd"))
	{
		cout << "=========== PFD ===========" << endl;
		abc::graph::cluster_pfd();
	}
}

// main
int main(int argc, char **argv)
{

	reader(argc, argv);
	
	return 0;
}
