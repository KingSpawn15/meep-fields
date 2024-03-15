#!/bin/bash



mpirun -np 10 python3.11 combined_pd_or.py 5
mpirun -np 10 python3.11 combined_pd_or.py 0.5


