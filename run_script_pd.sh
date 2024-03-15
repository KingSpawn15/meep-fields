#!/bin/bash

outdir='photodember'

mpirun -np 10 python3.11 combined_pd_or.py $outdir