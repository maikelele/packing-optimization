import numpy as np
import pygad
import random
import matplotlib.pyplot as plt
from math import pi

N = 10  # number of circles
square_size = 1.0 
r_min = 0.01  # Minimum radius
r_max = square_size / 2  # Maximum radius (half the square size)

# Fitness function to evaluate the packing
def fitness_function(ga_instance, solution, solution_idx):
    coords = solution[:-1].reshape((N, 2))
    r = solution[-1]

    # Radius must be positive and small enough to possibly fit
    if r <= 0 or r >= square_size / 2:
        return -1e6

    # Boundary check: all circles must be inside the square
    for cx, cy in coords:
        if (cx - r < 0 or cx + r > square_size or
            cy - r < 0 or cy + r > square_size):
            return -1e6

    # Overlap check: distance between centers must be at least 2r
    for i in range(N):
        for j in range(i + 1, N):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            distance_squared = dx * dx + dy * dy
            if distance_squared < (2 * r) ** 2:
                return -1e6

    return r  # Maximize radius → minimize -r


def create_random_solution():
    positions = np.random.uniform(0, square_size, (N, 2))
    radius_upper_bound = square_size / 2
    radius = np.random.uniform(r_min, radius_upper_bound, 1)
    return np.concatenate([positions.flatten(), radius])



# Function to mutate a solution
def on_mutation(ga_instance, offspring_population):
    for offspring in offspring_population:
        mutation_index = random.randint(0, len(offspring) - 1)
        
        if mutation_index < N * 2:
            offspring[mutation_index] = np.random.uniform(0, square_size)
        else:
            offspring[mutation_index] = np.random.uniform(r_min, r_max)
    
    return offspring_population


# Setup the genetic algorithm
num_generations = 5
population_size = 5
parent_selection_type = "rws"  # Roulette Wheel Selection
crossover_type = "uniform"
crossover_probability = 0.7
mutation_probability = 0.2
elitism = 1  # Corrected argument for pyGAD to retain the best solutions

initial_population = np.array([create_random_solution() for _ in range(population_size)])

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=2,
    fitness_func=fitness_function,
    sol_per_pop=population_size,
    num_genes=N * 2 + 1,
    gene_type=float,
    initial_population=initial_population,  # ✅ Use custom solutions here
    parent_selection_type=parent_selection_type,
    crossover_type=crossover_type,
    crossover_probability=crossover_probability,
    mutation_type="random",
    mutation_probability=mutation_probability,
    on_mutation=on_mutation,
    keep_elitism=elitism,
    save_best_solutions=True
)


# Run the genetic algorithm
ga_instance.run()

# Saving best solutiosn from each generation
best_solutions_per_generation = ga_instance.best_solutions

# Retrieve the best solution
best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()

# Decode the best solution
best_positions = best_solution[:-1].reshape((N, 2))  # Extract x, y positions
best_radius = best_solution[-1]  # Extract the shared radius

plt.figure(figsize=(8, 8))
for i in range(N):
    circle = plt.Circle((best_positions[i][0], best_positions[i][1]), best_radius, fill=False, edgecolor='green')
    plt.gca().add_patch(circle)

plt.xlim(0, square_size)
plt.ylim(0, square_size)
plt.gca().set_aspect('equal', adjustable='box')
plt.title(f"GA Optimal Packing of {N} Circles with Shared Radius in a Square")
plt.grid(True)
plt.show()
