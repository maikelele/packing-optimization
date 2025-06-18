import numpy as np
import pygad
import matplotlib.pyplot as plt
from math import pi
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for plotting scripts

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

    return r


def create_random_solution():
    positions = np.random.uniform(0, square_size, (N, 2))
    radius = np.random.uniform(r_max / 20, r_max / 8, 1) # Ensure radius is not too small or too large
    return np.concatenate([positions.flatten(), radius])


population_size = 50
num_generations = 10000
crossover_probability = 0.8
mutation_probability = [.8, .3]
best_coordinates = []
best_radii = []

fig, ax = plt.subplots(figsize=(8, 8))
def on_start(ga_instance):
    print("Starting the genetic algorithm...")

def on_fitness(ga_instance, population_fitness):
    pass

def on_parents(ga_instance: pygad.GA, parents):
    pass

def on_crossover(ga_instance, offspring_crossover):
    print(f"Crossover occurred: {offspring_crossover}")

def on_mutation(ga_instance, offspring_mutation):
    print(f"Mutation occurred: {offspring_mutation}")

def on_stop(ga_instance, last_population_fitness):
    print("Stopping the genetic algorithm...")
    print(f"Last population fitness: {last_population_fitness}")

initial_population = np.array([create_random_solution() for _ in range(population_size)])


def on_generation(ga_instance):
    # This function is called at the end of each generation
    print(f"Generation {ga_instance.generations_completed} - Best fitness: {ga_instance.best_solution()[1]}")
    best_coordinates.append(ga_instance.best_solution()[0][:-1].reshape((N, 2)))
    best_radii.append(ga_instance.best_solution()[0][-1])

    def plot():
        ax.cla()
        ax.set_xlim(0, square_size)
        ax.set_ylim(0, square_size)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        for i in range(N):
            circle = plt.Circle(best_coordinates[-1][i], best_radii[-1], color='green', fill=False)
            ax.add_patch(circle)
        ax.set_title(f"Generation {ga_instance.generations_completed} - Best fitness: {ga_instance.best_solution()[1]}")
        plt.pause(np.nextafter(0, 1))  # Pause to update the plot

    plot()

# Initialize the genetic algorithm
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=10,
    fitness_func=fitness_function,
    initial_population=initial_population,
    parallel_processing=12, 
    parent_selection_type='sss',
    crossover_type='single_point',
    crossover_probability=crossover_probability,
    mutation_type='adaptive',
    mutation_probability=mutation_probability,
    gene_space = [{'low': 0.1, 'high': square_size - 0.1}] * 20 + [{'low': 0.085, 'high': 0.15}],
    # on_start=on_start,
    # on_fitness=on_fitness,
    # on_parents=on_parents,
    # on_crossover=on_crossover,
    # on_mutation=on_mutation,
    on_generation=on_generation,
    # on_stop=on_stop,
    keep_elitism=10,
)

# Run the genetic algorithm
ga_instance.run()
