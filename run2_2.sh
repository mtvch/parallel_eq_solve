#!/bin/bash

#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=2:mpiprocs=2:mem=2000m

cd $PBS_O_WORKDIR
MPI_NP=$(wc -l $PBS_NODEFILE | awk '{ print $1 }')

mpicc main2.c -o main2 -lm -O2 -std=gnu99

mpirun -machinefile $PBS_NODEFILE -np $MPI_NP ./main2 144
