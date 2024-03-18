#!/bin/bash

mpirun -np 12 python3.11 combined_pd_or.py 5 0.6
mpirun -np 12 python3.11 combined_pd_or.py 5 0.7
mpirun -np 12 python3.11 combined_pd_or.py 5 0.8
mpirun -np 12 python3.11 combined_pd_or.py 10 0.6
mpirun -np 12 python3.11 combined_pd_or.py 10 0.7
mpirun -np 12 python3.11 combined_pd_or.py 10 0.8
