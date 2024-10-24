#!/bin/bash

# mpirun -np 10 python3.11 combined_pd_or.py 0.10 0.5 50
# mpirun -np 10 python3.11 combined_pd_or.py 0.17 0.5 50
# mpirun -np 10 python3.11 combined_pd_or.py 0.30 0.5 50
# mpirun -np 10 python3.11 combined_pd_or.py 0.52 0.5 50
# mpirun -np 10 python3.11 combined_pd_or.py 1.00 0.5 50
# mpirun -np 10 python3.11 combined_pd_or.py 1.73 0.5 50
# mpirun -np 10 python3.11 combined_pd_or.py 3.00 0.5 50
# mpirun -np 10 python3.11 combined_pd_or.py 5.19 0.5 50
# mpirun -np 10 python3.11 combined_pd_or.py 10.00 0.5 50
# mpirun -np 10 python3.11 combined_pd_or.py 17.30 0.5 50
# mpirun -np 10 python3.11 combined_pd_or.py 30.00 0.5 50

# mpirun -np 10 python3.11 combined_pd_or.py 5.0 0.5 250
# mpirun -np 10 python3.11 combined_pd_or.py 5.0 0.5 300
# mpirun -np 10 python3.11 combined_pd_or.py 5.0 0.5 150
# mpirun -np 10 python3.11 combined_pd_or.py 5.0 0.5 350
# mpirun -np 10 python3.11 combined_pd_or.py 5.0 0.5 70
# mpirun -np 10 python3.11 combined_pd_or.py 5.0 0.5 90
# mpirun -np 10 python3.11 combined_pd_or.py 5.0 0.5 110
# mpirun -np 10 python3.11 combined_pd_or.py 5.0 0.5 130
# mpirun -np 10 python3.11 combined_pd_or.py 5.0 0.5 50
# mpirun -np 10 python3.11 combined_pd_or.py 5.0 0.5 170
# mpirun -np 10 python3.11 combined_pd_or.py 5.0 0.5 190

# mpirun -np 10 python3.11 combined_pd_or.py 10 0.5 50


# mpirun -np 8 python3.11 lateral_pd.py 10 0.5 50
# mpirun -np 10 python3.11 lateral_pd.py 10 0.5 100
# mpirun -np 10 python3.11 lateral_pd.py 10 0.5 200

mpirun -np 1 python3.11 photodember_driftdiffusion.py 10 0.5 50 0.1
mpirun -np 1 python3.11 photodember_driftdiffusion.py 10 0.5 50 0.05
mpirun -np 1 python3.11 photodember_driftdiffusion.py 10 0.5 50 0.02