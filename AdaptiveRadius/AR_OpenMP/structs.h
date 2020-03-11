#ifndef _STRUCTS_H
#define _STRUCTS_H

typedef struct 
{
  int clusterId;
  double* features;
} objectInit_t;

typedef struct
{
  int x;
  int y;
} object_t;

typedef struct
{
  int x;
  int y;
  int objectId;
  long seed;
} ant_t;

#endif








