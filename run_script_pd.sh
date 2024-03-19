#!/bin/bash

mpirun -np 12 python3.11 combined_pd_or.py 0.10 0.5
mpirun -np 12 python3.11 combined_pd_or.py 0.17 0.5
mpirun -np 12 python3.11 combined_pd_or.py 0.30 0.5
mpirun -np 12 python3.11 combined_pd_or.py 0.52 0.5
mpirun -np 12 python3.11 combined_pd_or.py 1.00 0.5
mpirun -np 12 python3.11 combined_pd_or.py 1.73 0.5
mpirun -np 12 python3.11 combined_pd_or.py 3.00 0.5
mpirun -np 12 python3.11 combined_pd_or.py 5.19 0.5
mpirun -np 12 python3.11 combined_pd_or.py 10.00 0.5
mpirun -np 12 python3.11 combined_pd_or.py 17.30 0.5
mpirun -np 12 python3.11 combined_pd_or.py 30.00 0.5
