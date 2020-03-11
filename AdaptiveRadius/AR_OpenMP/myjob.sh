#!/bin/bash
#$ -cwd
#$ -o /dev/null
#$ -e /dev/null
# Set the number of threads here:
#$ -pe smp 16
# Set the number of processors (if different from the above) here:
setenv OMP_NUM_THREADS $NSLOTS
# Change the name of the execuatable here:
./OMP_AntBroodClustering > output.txt