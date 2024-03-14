#!/bin/bash

outdir='photodember'

mpirun -np 50 python3.11 combined_pd_or.py $outdir