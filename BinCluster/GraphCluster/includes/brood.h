#ifndef _BROOD_H
#define _BROOD_H

#include "globals.h"

class Node
{
public:
	Node();
	~Node();

	int objectId;
	Node* next;
};


class Bin
{
public:
	Bin();
	~Bin();

	// access to an random object in this bin
	int rand_obj();
	// insert
	void insert(int id);
	// delete
	void remove(int id);

	int size;
	int id;
	Node* head;
	Bin* next;
};


class Brood
{
public:
	Brood(int n);
	~Brood();

	// create an new empty bin, return bin id
	int create_bin();
	// remove bin by id
	void delete_bin(int id);
	// get bin by id
	Bin* get_bin(int id);
	// access to an random bin
	Bin* rand_bin();
	// print
	void print();

	int size;
	int idCount;
	Bin* head;
};


#endif