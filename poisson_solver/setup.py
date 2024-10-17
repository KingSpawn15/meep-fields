from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='FastPoissonSolver',
    ext_modules=cythonize("poisson_solver/fast_poisson_solver.pyx"),  # Name of your Cython file
    include_dirs=[np.get_include()],  # Include numpy headers
)