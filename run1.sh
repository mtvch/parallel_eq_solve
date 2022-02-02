#!/bin/bash

#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=1:mem=1000m

cd $PBS_O_WORKDIR

gcc main1.c -o main1 -lm -O2 -std=gnu99

./main1 144
