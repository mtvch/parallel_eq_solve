#!/bin/bash

#PBS -l walltime=00:05:00
#PBS -l select=2:ncpus=8:mpiprocs=8:mem=3000m

cd $PBS_O_WORKDIR
MPI_NP=$(wc -l $PBS_NODEFILE | awk '{ print $1 }')

mpiicc main3.c -o main3 -lm -O2 -std=gnu99

mpirun -trace -machinefile $PBS_NODEFILE -np $MPI_NP ./main3 144

