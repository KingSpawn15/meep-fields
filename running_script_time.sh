#!/bin/bash

mpirun -np 10 python3.11 rectification_organized.py 25
mpirun -np 10 python3.11 rectification_organized.py 30
mpirun -np 10 python3.11 rectification_organized.py 50
mpirun -np 10 python3.11 rectification_organized.py 75
mpirun -np 10 python3.11 rectification_organized.py 100
