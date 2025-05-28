import numpy as np
import matplotlib.pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO
from math import pi

# Parameters
N = 10  # number of circles
square_size = 1.0

# Define bounds for x, y coordinates and radius
lower_bounds = np.array([0.0] * (2 * N) + [0.01])
upper_bounds = np.array([square_size] * (2 * N) + [square_size / 2])
bounds = (lower_bounds, upper_bounds)

# Define the objective function to maximize radius (minimize -radius)
def pso_objective(x):
    fitness = []
    for particle in x:
        coords = particle[:-1].reshape((N, 2))
        r = particle[-1]
        valid = True

        # Check if all circles are within bounds
        for cx, cy in coords:
            if not (r <= cx <= square_size - r and r <= cy <= square_size - r):
                valid = False
                break

        # Check for overlaps
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = coords[i][0] - coords[j][0]
                    dy = coords[i][1] - coords[j][1]
                    if dx**2 + dy**2 < (2 * r)**2:
                        valid = False
                        break
                if not valid:
                    break

        # Fitness value: maximize r (so minimize -r), penalize invalid configurations
        fitness.append(-r if valid else 1e6)

    return np.array(fitness)

# Initialize and run PSO
optimizer = GlobalBestPSO(n_particles=500, dimensions=2 * N + 1, options={'c1': 2, 'c2': 1.5, 'w': 0.55}, bounds=bounds)
cost, pos = optimizer.optimize(pso_objective, iters=20000)

# Extract optimized values
optimized_coords = pos[:-1].reshape((N, 2))
optimized_radius = pos[-1]

# Compute metrics
max_distance = np.max(np.linalg.norm(optimized_coords - np.array([square_size/2, square_size/2]), axis=1))
ratio = 1 / optimized_radius
density = N * pi * optimized_radius**2 / square_size**2

# Plot result
fig, ax = plt.subplots()
ax.set_aspect('equal')
for (x, y) in optimized_coords:
    circle = plt.Circle((x, y), optimized_radius, fill=False, edgecolor='green')
    ax.add_patch(circle)
plt.xlim(0, square_size)
plt.ylim(0, square_size)
plt.title(f"PSO Optimal Packing of {N} Circles in a Square")
plt.grid(True)
plt.show()

# Output result
import pandas as pd
summary = pd.DataFrame({
    "N": [N],
    "radius": [optimized_radius],
    "distance": [max_distance],
    "ratio": [ratio],
    "density": [density]
})
print(summary)

