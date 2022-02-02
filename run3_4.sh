#!/bin/bash

#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=4:mpiprocs=4:mem=2000m

cd $PBS_O_WORKDIR
MPI_NP=$(wc -l $PBS_NODEFILE | awk '{ print $1 }')

mpirun -machinefile $PBS_NODEFILE -np $MPI_NP ./main3 144
