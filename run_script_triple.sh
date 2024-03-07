#!/bin/bash

outdir='triple_test_ez'

# mpirun -np 50 python3.11 rectification_triple.py 30 $outdir
mpirun -np 50 python3.11 rectification_triple.py 50 $outdir
# mpirun -np 50 python3.11 rectification_triple.py 75 $outdir
# mpirun -np 50 python3.11 rectification_triple.py 100 $outdir
