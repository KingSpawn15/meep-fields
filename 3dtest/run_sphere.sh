#!/bin/bash

# mpirun -np 10 python3.11 sphere.py;

# h5tovtk -o epsilon.vtk structure_demo-eps-000000.00.h5;
h5tovtk -o epsilon.vtk sphere-eps-000000.00.h5;

mayavi2 -d epsilon.vtk -m IsoSurface &> /dev/null &