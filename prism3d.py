import meep as mp
import numpy as np
import math

cell_size = mp.Vector3(2,2,2)

vertices = [mp.Vector3(-1,0),
mp.Vector3(-0.5,math.sqrt(3)/2),
mp.Vector3(0.5,math.sqrt(3)/2),
mp.Vector3(1,0),
mp.Vector3(0.5,-math.sqrt(3)/2),
mp.Vector3(-0.5,-math.sqrt(3)/2)]

geometry = [mp.Prism(vertices, height=1.0, material=mp.Medium(index=3.5)),
mp.Cone(radius=1.0, radius2=0.1, height=2.0, material=mp.air)]

sim = mp.Simulation(resolution=50,
cell_size=cell_size,
geometry=geometry)

sim.plot3D()