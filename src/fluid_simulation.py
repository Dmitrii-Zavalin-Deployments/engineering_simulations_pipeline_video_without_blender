import os
import json
import numpy as np
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.sph.scheme import WCSPHScheme

# ✅ Define Fluid Simulation Class Using PySPH
class FluidSimulation(Application):
    def __init__(self):
        super().__init__()  # Call the __init__ of the parent class
        self.scheme = None  # Initialize scheme to None
        self.initialize()

    def initialize(self):
        """Setup fluid domain, particles, and physics properties."""
        self.create_particles()
        self.scheme = WCSPHScheme(
            fluids=['fluid'], solids=['boundary'],  # ✅ Added solids argument
            dim=2, rho0=1000, c0=10, h0=1.2,
            gamma=7.0, alpha=0.1, beta=0.1,
            hdx=1.3
        )

    def create_particles(self):
        """Define water particle positions and velocities."""
        fluid_particles = get_particle_array(name='fluid', dx=0.1, dy=0.1)
        boundary_particles = get_particle_array(name='boundary', dx=0.1, dy=0.1)

        # ✅ Ensure particles have initial velocity in the x-direction
        fluid_particles.u[:] = 15.0  # Apply strong horizontal velocity

        self.particles = {'fluid': fluid_particles, 'boundary': boundary_particles}

    def run(self):
        """Execute the solver and compute fluid motion."""
        self.scheme.configure_solver(dt=0.01, tf=3.0)
        solver = self.scheme.get_solver()
        solver.dump_output(output_at_times=[0.0, 1.0, 2.0, 3.0])
        solver.solve(self.particles)
        return self.particles

# ✅ Execute Fluid Simulation & Store Results
simulation = FluidSimulation()
fluid_data = simulation.run()

# ✅ Convert Particle Positions to JSON Format
fluid_output = {
    "frames": [],
    "fluid_domain": {
        "dimensions": [50, 10, 5],  # Fluid domain size
        "gravity": [0, 0, 0]  # Gravity disabled
    }
}

num_frames = len(np.atleast_1d(fluid_data['fluid'].x))

for frame_idx in range(num_frames):
    frame_data = {
        "particles_x": fluid_data['fluid'].x.tolist(),
        "particles_y": fluid_data['fluid'].y.tolist(),
        "velocity_vectors": fluid_data['fluid'].u.tolist()
    }
    fluid_output["frames"].append(frame_data)

# ✅ Save Fluid Simulation Results
output_path = "data/testing-input-output/fluid_dynamics_animation.json"

with open(output_path, "w") as f:
    json.dump(fluid_output, f, indent=4)

print(f"✅ Fluid simulation setup complete! Data saved as '{output_path}'.")


