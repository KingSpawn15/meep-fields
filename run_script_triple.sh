#!/bin/bash

outdir='rectification_pulse_time'

# mpirun -np 8 python3.11 rectification_triple.py 50 $outdir 0.4
# mpirun -np 8 python3.11 rectification_triple.py 70 $outdir 0.4
# mpirun -np 8 python3.11 rectification_triple.py 90 $outdir 0.4
# mpirun -np 8 python3.11 rectification_triple.py 110 $outdir 0.4
# mpirun -np 8 python3.11 rectification_triple.py 130 $outdir 0.4
# mpirun -np 8 python3.11 rectification_triple.py 150 $outdir 0.4
# mpirun -np 8 python3.11 rectification_triple.py 170 $outdir 0.4
# mpirun -np 8 python3.11 rectification_triple.py 190 $outdir 0.4
# mpirun -np 8 python3.11 rectification_triple.py 250 $outdir 0.4
# mpirun -np 8 python3.11 rectification_triple.py 300 $outdir 0.4
# mpirun -np 8 python3.11 rectification_triple.py 350 $outdir 0.4

# mpirun -np 10 python3.11 combined_pd_or.py 0.10 0.6 50
# mpirun -np 10 python3.11 combined_pd_or.py 0.17 0.6 50
# mpirun -np 10 python3.11 combined_pd_or.py 0.30 0.6 50
# mpirun -np 10 python3.11 combined_pd_or.py 0.52 0.6 50
# mpirun -np 10 python3.11 combined_pd_or.py 1.00 0.6 50
# mpirun -np 10 python3.11 combined_pd_or.py 1.73 0.6 50
# mpirun -np 10 python3.11 combined_pd_or.py 3.00 0.6 50
# mpirun -np 10 python3.11 combined_pd_or.py 5.19 0.6 50
# mpirun -np 10 python3.11 combined_pd_or.py 10.00 0.6 50
# mpirun -np 10 python3.11 combined_pd_or.py 17.30 0.6 50
# mpirun -np 10 python3.11 combined_pd_or.py 30.00 0.6 50

# mpirun -np 10 python3.11 combined_pd_or.py 10.00 0.6 50
mpirun -np 10 python3.11 combined_pd_or.py 10.00 0.6 70
mpirun -np 10 python3.11 combined_pd_or.py 10.00 0.6 90
mpirun -np 10 python3.11 combined_pd_or.py 10.00 0.6 110
mpirun -np 10 python3.11 combined_pd_or.py 10.00 0.6 130
mpirun -np 10 python3.11 combined_pd_or.py 10.00 0.6 150
mpirun -np 10 python3.11 combined_pd_or.py 10.00 0.6 170
mpirun -np 10 python3.11 combined_pd_or.py 10.00 0.6 190
mpirun -np 10 python3.11 combined_pd_or.py 10.00 0.6 250
mpirun -np 10 python3.11 combined_pd_or.py 10.00 0.6 300
mpirun -np 10 python3.11 combined_pd_or.py 10.00 0.6 350

