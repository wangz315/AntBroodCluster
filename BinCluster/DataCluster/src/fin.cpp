// header files
// local headers
#include "fin.h"


void print_cluster()
{
	string colors[11] = {"\033[0m", "\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[35m", "\033[36m", "\033[37m", "\033[38m", "\033[39m", "\033[40m"};

	Bin* bin = brood->head;
	for(int i = 0; i < brood->size; i++)
	{
		cout << "Bin " << bin->id << " : ";
		Node* node = bin->head;
		for(int j = 0; j < bin->size; j++)
		{
			int clusterId = features[node->objectId*numObjects+featureLength];
			cout << clusterId;
			node = node->next;
		}
		cout << endl;

		bin = bin->next;
	}
}



void fin()
{
	print_cluster();
}