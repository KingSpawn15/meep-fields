#!/bin/bash

angle1=30
angle2=150
outdir='set14'

mpirun -np 50 python3.11 rectification_organized.py 30 $angle1 $angle2 $outdir
mpirun -np 50 python3.11 rectification_organized.py 50 $angle1 $angle2 $outdir
mpirun -np 50 python3.11 rectification_organized.py 75 $angle1 $angle2 $outdir
mpirun -np 50 python3.11 rectification_organized.py 100 $angle1 $angle2 $outdir


