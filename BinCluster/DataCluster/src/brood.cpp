#include "globals.h"

Node::Node()
{
	objectId = -1;
	next = NULL;
}

Node::~Node(){}


Bin::Bin()
{
	size = 0;
	id = -1;
	head = NULL;
	next = NULL;
}

Bin::~Bin(){}

void Bin::insert(int id)
{
	Node* node = new Node();
	node->objectId = id;

	if(head == NULL)
	{
		head = node;
	}
	else
	{
		node->next = head;
		head = node;
	}

	size++;
}


void Bin::remove(int id)
{
	if(size == 1)
	{
		if(head->objectId == id)
		{
			head = NULL;
			size = 0;
		}
	}
	else
	{
		Node* curr = head;
		Node* prev = NULL;

		while(curr != NULL)
		{
			if(curr->objectId == id)
			{
				if(curr == head)
				{
					head = head->next;
					size--;
					break;
				}
				else
				{
					prev->next = curr->next;
					size--;
					break;
				}
			}

			prev = curr;
			curr = curr->next;
		}
	}
}


int Bin::rand_obj()
{
	int index = rand()%size;
	Node* node = head;

	for(int i = 0; i < index; i++)
	{
		node = node->next;
	}

	return node->objectId;
}


Brood::Brood(int n)
{
	size = n;
	idCount = n+1;

	head = new Bin();
	head->id = 0;
	head->insert(0);

	for(int i = 1; i < n; i++)
	{
		Bin* bin = new Bin();
		bin->id = i;
		bin->insert(i);

		bin->next = head;
		head = bin;
	}
}


Brood::~Brood(){}


int Brood::create_bin()
{
	Bin* bin = new Bin();
	bin->id = idCount;
	idCount++;
	size++;

	bin->next = head;
	head = bin;

	return bin->id;
}


void Brood::delete_bin(int id)
{
	Bin* prev = NULL;
	Bin* curr = head;
	for(int i = 0; i < size; i++)
	{
		if(curr->id == id)
		{
			if(curr == head)
			{
				head = head->next;
				size--;
				break;
			}
			else
			{
				prev->next = curr->next;
				size--;
				break;
			}
		}

		prev = curr;
		curr = curr->next;
	}
}


Bin* Brood::get_bin(int id)
{
	Bin* bin = head;

	for(int i = 0; i < size; i++)
	{
		if(bin->id == id)
		{
			return bin;
		}

		bin = bin->next;
	}

	return NULL;
}

Bin* Brood::rand_bin()
{
	int index = rand()%size;
	Bin* bin = head;

	for(int i = 0; i < index; i++)
	{
		bin = bin->next;
	}

	return bin;
}


void Brood::print()
{
	Bin* bin = head;
	for(int i = 0; i < size; i++)
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