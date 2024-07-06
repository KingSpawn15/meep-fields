import meep as mp

# Set resolution
resolution = 50

# Set geometry-lattice
lattice = mp.Lattice(size=mp.Vector3(3, 3, 3))

# Set geometry
geometry = [mp.Sphere(radius=1, material=mp.Medium(index=3.5), center=mp.Vector3(0, 0, 0)),
            mp.Cone(radius=0.8, radius2=0.1, height=2, material=mp.air, center=mp.Vector3(0, 0, 0))]

# Initialize simulation
sim = mp.Simulation(cell_size=mp.Vector3(3, 3, 3),
                    boundary_layers=[mp.PML(1.0)],
                    geometry=geometry,
                    resolution=resolution)

# Output epsilon
sim.run(mp.at_beginning(mp.output_epsilon),
        until=0)

# Exit
exit()