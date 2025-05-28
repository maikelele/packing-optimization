import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import pi
import pandas as pd

# Parameters
circle_number = 10
square_size = 1.0

# Objective: Maximize the radius (minimize the negative radius)
def objective(x):
    r = x[-1]
    return -r


# Constraints:
# 1. Circles must be within the square
# 2. Circles must not overlap
def constraints(x):
    cons = []
    r = x[-1]
    coords = x[:-1].reshape((circle_number, 2))

    # Circles within the square
    for cx, cy in coords:
        cons.append(cx - r)
        cons.append(cy - r)
        cons.append(square_size - (cx + r))
        cons.append(square_size - (cy + r))

    # Circles do not overlap
    for i in range(circle_number):
        for j in range(i + 1, circle_number):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist_sq = dx ** 2 + dy ** 2
            cons.append(dist_sq - (2 * r) ** 2)

    return cons


# Initial guess
initial_positions = np.random.rand(circle_number * 2) * square_size
initial_radius = 0.05
x0 = np.append(initial_positions, initial_radius)

# Define constraint dictionary for scipy minimize
cons = {'type': 'ineq', 'fun': lambda x: np.array(constraints(x))}

# Run optimization
result = minimize(objective, x0, constraints=cons, method='SLSQP', options={'disp': True, 'maxiter': 1000})

# Extract results
optimized_coords = result.x[:-1].reshape((circle_number, 2))
optimized_radius = result.x[-1]

# Compute metrics
max_distance = np.max(np.linalg.norm(optimized_coords - np.array([square_size / 2, square_size / 2]), axis=1))
ratio = 1 / optimized_radius
density = circle_number * pi * optimized_radius ** 2 / square_size ** 2

# Plot the result
fig, ax = plt.subplots()
ax.set_aspect('equal')
for (x, y) in optimized_coords:
    circle = plt.Circle((x, y), optimized_radius, fill=False, edgecolor='blue')
    ax.add_patch(circle)
plt.xlim(0, square_size)
plt.ylim(0, square_size)
plt.title(f"Optimal Packing of {circle_number} Circles in a Square")
plt.grid(True)
plt.show()

# Summary of results
summary = pd.DataFrame({
    "Number of circles": [circle_number],
    "radius": [optimized_radius],
    "distance": [max_distance],
    "ratio": [ratio],
    "density": [density]
})
print(summary)