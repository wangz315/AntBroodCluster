#ifndef _STRUCTS_H
#define _STRUCTS_H

// only used for data files
typedef struct
{
  int clusterId;
  double* features;
} objectInfo_t;

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
  long seed; // seed is used for random
} ant_t;

#endif








