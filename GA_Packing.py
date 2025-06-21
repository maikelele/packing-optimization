import os
import sys
from math import pi
import random

import numpy as np
import pygad
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for plotting scripts


N = 10  # number of circles
square_size = 1.0 
r_min = 0.01  
r_max = square_size / 2  
population_size = 100
num_generations = 20_000
crossover_probability = 0.7
mutation_probability = 1
best_coordinates = []
best_radii = []
num_parents_mating = 50
keep_elitism = 30
fitness_debugging = False

# Fitness function to evaluate the packing
def fitness_function(ga_instance, solution, solution_idx):
    global fitness_debugging
    coords = solution[:-1].reshape((N, 2))
    r = solution[-1]

    # Radius must be positive and small enough to possibly fit
    if r <= 0 or r >= square_size / 2:
        if fitness_debugging:
            print(f"Invalid radius: {r:.4f}")
        return -1e6

    # Boundary check: all circles must be inside the square
    for cx, cy in coords:
        if (cx - r < 0 or cx + r > square_size or
            cy - r < 0 or cy + r > square_size):
            if fitness_debugging:
                print(f"Circle at ({cx:.4f}, {cy:.4f}) with radius {r:.4f} is out of bounds.")
            return -1e6

    # Overlap check: distance between centers must be at least 2r
    for i in range(N):
        for j in range(i + 1, N):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            distance_squared = dx * dx + dy * dy
            if distance_squared < (2 * r) ** 2:
                if fitness_debugging:
                    print(f"Overlap detected between circles {i} and {j}. Distance squared: {distance_squared:.4f}, Required: {(2 * r) ** 2:.4f}")
                return -1e6

    fitness_debugging = False

    return r


def create_random_solution():
    positions = np.random.uniform(0, square_size, (N, 2))
    radius = np.random.uniform(r_max / 20, r_max / 8, 1) # Ensure radius is not too small or too large
    return np.concatenate([positions.flatten(), radius])


def on_fitness(ga_instance: pygad.GA, population_fitness):
    print(f"Best fitness values for generation {ga_instance.generations_completed}: {ga_instance.best_solution()[1]}")
    for idx, el in enumerate(population_fitness):
        if idx % 25 == 0 and idx > 0:
            print()
        if el < 0:
            print(f"{'-INF':6s}", end='\t')
        else:
            print(f"{el:5.2f}", end='\t')
    print('\n')


def on_parents(ga_instance: pygad.GA, parents):
    global fitness_debugging
    print(f"Parents selected for generation {ga_instance.generations_completed}")
    for idx, parent in enumerate(parents):
        coords = parent[:-1].reshape((N, 2))
        coords = [f"({coord[0]:.2f}, {coord[1]:.2f})" for coord in coords]
        coords = ', '.join(coords)
        r = parent[-1]
        fitness_debugging = True
        print(f"Parent {idx}: {coords}, Radius: {r:.2f}, Fitness: {fitness_function(ga_instance, parent, 0):.4f}")
    print('\n')


def on_crossover(ga_instance: pygad.GA, offspring_crossover):
    global fitness_debugging
    print(f"Crossover solutions for generation {ga_instance.generations_completed}")
    for idx, solution in enumerate(offspring_crossover):
        coords = solution[:-1].reshape((N, 2))
        coords = [f"({coord[0]:.2f}, {coord[1]:.2f})" for coord in coords]
        coords = ', '.join(coords)
        r = solution[-1]
        fitness_debugging = True
        print(f"Cross sol {idx}: {coords}, Radius: {r:.2f}, Fitness: {fitness_function(ga_instance, solution, 0):.4f}")
    print('\n')


def on_mutation(ga_instance, offspring_mutation):
    global fitness_debugging
    print(f"Mutation solutions for generation {ga_instance.generations_completed}")
    for idx, solution in enumerate(offspring_mutation):
        coords = solution[:-1].reshape((N, 2))
        coords = [f"({coord[0]:.2f}, {coord[1]:.2f})" for coord in coords]
        coords = ', '.join(coords)
        r = solution[-1]
        fitness_debugging = True
        print(f"Mut sol {idx}: {coords}, Radius: {r:.2f}, Fitness: {fitness_function(ga_instance, solution, 0):.4f}")
    print('\n')


fig, ax = plt.subplots(figsize=(8, 8))
def on_generation(ga_instance: pygad.GA):
    def plot():
        best_coordinates.append(ga_instance.best_solution()[0][:-1].reshape((N, 2)))
        best_radii.append(ga_instance.best_solution()[0][-1])
        ax.cla()
        
        
        for i in range(N):
            circle = plt.Circle(best_coordinates[-1][i], best_radii[-1], color='green', alpha=0.5)
            ax.add_patch(circle)
        ax.set_title(f"Generation {ga_instance.generations_completed} - Best fitness: {ga_instance.best_solution()[1]}")
        plt.pause(np.nextafter(0, 1))  # Pause to update the plot

    if plotting_enabled:
        plot()


def custom_crossover(parents: np.ndarray, offspring_size: tuple, ga_instance: pygad.GA) -> np.ndarray:
    """
    Custom crossover function for circle packing.
    Combines circle positions and radius from two parents to create offspring.
    """
    offspring = np.empty(offspring_size)

    for k in range(offspring_size[0]):
        # Select two random parents
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]

        # Split genes into positions and radius
        positions1, radius1 = parent1[:-1], parent1[-1]
        positions2, radius2 = parent2[:-1], parent2[-1]

        # Get 2 circles from each parent
        child_positions = np.empty_like(positions1)
        chosen_indices1 = random.sample(np.arange(N).tolist(), 2)
        chosen_indices2 = random.sample(np.delete(np.arange(N), chosen_indices1).tolist(), 2)
        for i in chosen_indices1:
            child_positions[i * 2] = positions1[i * 2]
            child_positions[i * 2 + 1] = positions1[i * 2 + 1]
        for i in chosen_indices2:
            child_positions[i * 2] = positions2[i * 2]
            child_positions[i * 2 + 1] = positions2[i * 2 + 1]

        # Combine radius (weighted average)
        child_radius = (radius1 + radius2) / 2

        # Create the offspring
        offspring[k, :-1] = child_positions
        offspring[k, -1] = child_radius

        # Validate the offspring
        offspring[k, :-1] = np.clip(offspring[k, :-1], 0, square_size)  # Ensure positions are inside the square
        offspring[k, -1] = np.clip(offspring[k, -1], r_min, r_max)      # Ensure radius is within valid range

    return offspring


def custom_mutation(last_generation_offspring_crossover: np.ndarray, ga_instance: pygad.GA):
    """
    Custom mutation function which applies a small mutation step to each gene in the offspring
    """
    mutation_step = 0.01  # Small step for mutation
    near_optimal_mutation_step = mutation_step / 10  # Smaller step when near optimal solution
    near_optimal_threshold = 1.1  # Threshold to consider a solution near optimal
    for offspring in last_generation_offspring_crossover:
        # Find a least dense quarter of the square and mutate there
        halves = ['left', 'right', 'top', 'bottom']
        circles_in_halves = {key: list() for key in halves}
        for i in range(N):
            x, y = offspring[i * 2], offspring[i * 2 + 1]
            if x < square_size / 2:
                circles_in_halves['left'].append(i)
            elif x >= square_size / 2:
                circles_in_halves['right'].append(i)
            if y < square_size / 2:
                circles_in_halves['bottom'].append(i)
            elif y >= square_size / 2:
                circles_in_halves['top'].append(i)

        least_dense_half = min(circles_in_halves, key=lambda k: len(circles_in_halves[k]))

        if fitness_function(0, offspring, 0) > near_optimal_threshold:
            if circles_in_halves[least_dense_half] is not None and len(circles_in_halves[least_dense_half]) > 0:
                # Mutate only the circles in the least dense half with a smaller step
                for i in circles_in_halves[least_dense_half]:
                    offspring[i * 2] += np.random.uniform(-near_optimal_mutation_step, near_optimal_mutation_step)
                    offspring[i * 2 + 1] += np.random.uniform(-near_optimal_mutation_step, near_optimal_mutation_step)
                    # Ensure positions are inside the square
                    offspring[i * 2] = np.clip(offspring[i * 2], 0, square_size)
                    offspring[i * 2 + 1] = np.clip(offspring[i * 2 + 1], 0, square_size)
            else:
                # Mutate all circles with a smaller step
                for i in range(N):
                    offspring[i * 2] += np.random.uniform(-near_optimal_mutation_step, near_optimal_mutation_step)
                    offspring[i * 2 + 1] += np.random.uniform(-near_optimal_mutation_step, near_optimal_mutation_step)
                    # Ensure positions are inside the square
                    offspring[i * 2] = np.clip(offspring[i * 2], 0, square_size)
                    offspring[i * 2 + 1] = np.clip(offspring[i * 2 + 1], 0, square_size)
        else:
            if circles_in_halves[least_dense_half] is not None and len(circles_in_halves[least_dense_half]) > 0:
                # Mutate only circles in the least dense half
                for i in circles_in_halves[least_dense_half]:
                    offspring[i * 2] += np.random.uniform(-mutation_step, mutation_step)
                    offspring[i * 2 + 1] += np.random.uniform(-mutation_step, mutation_step)
                    # Ensure positions are inside the square
                    offspring[i * 2] = np.clip(offspring[i * 2], 0, square_size)
                    offspring[i * 2 + 1] = np.clip(offspring[i * 2 + 1], 0, square_size)
                offspring[-1] += np.random.uniform(0, mutation_step)
            else:
                # Mutate all circles
                for i in range(N):
                    offspring[i * 2] += np.random.uniform(-mutation_step, mutation_step)
                    offspring[i * 2 + 1] += np.random.uniform(-mutation_step, mutation_step)
                    # Ensure positions are inside the square
                    offspring[i * 2] = np.clip(offspring[i * 2], 0, square_size)  
                    offspring[i * 2 + 1] = np.clip(offspring[i * 2 + 1], 0, square_size)  
                offspring[-1] += np.random.uniform(0, mutation_step)

        # Update radius by a normal value if the fitness is positive
        if fitness_function(0, offspring, 0) > 0:
            if fitness_function(0, offspring, 0) > near_optimal_threshold:
                offspring[-1] += random.uniform(-near_optimal_mutation_step, near_optimal_mutation_step)
            else:
                offspring[-1] += random.uniform(-mutation_step, mutation_step)
        # Else reduce the radius to and change positions of circles to avoid local minimum
        else:
            offspring[-1] -= random.uniform(0, 3 * mutation_step) # to avoid getting stuck in a local minimum
            for i in range(N):
                offspring[i * 2] += random.uniform(-10 * mutation_step, 10 * mutation_step) # to avoid getting stuck in a local minimum
                offspring[i * 2] = np.clip(offspring[i * 2], 0, square_size)  # Ensure positions are inside the square
                offspring[i * 2 + 1] += random.uniform(-10 * mutation_step, 10 * mutation_step) 
                offspring[i * 2 + 1] = np.clip(offspring[i * 2 + 1], 0, square_size)  

    return last_generation_offspring_crossover

initial_population = np.array([create_random_solution() for _ in range(population_size)])

# Initialize the genetic algorithm
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    initial_population=initial_population,
    parallel_processing=12, 
    parent_selection_type='sss',
    crossover_type=custom_crossover,
    crossover_probability=crossover_probability,
    mutation_type=custom_mutation, # because of the specificity of the problem custom mutation is needed
    mutation_probability=mutation_probability,
    gene_space = [{'low': 0.1, 'high': square_size - 0.1}] * 20 + [{'low': 0.085, 'high': 0.15}],
    on_fitness=on_fitness,
    on_parents=on_parents,
    on_crossover=on_crossover,
    on_mutation=on_mutation,
    on_generation=on_generation,
    keep_elitism=keep_elitism,
    allow_duplicate_genes=True,
)

# Run the genetic algorithm
if __name__ == "__main__":
    global plotting_enabled
    if len(sys.argv) > 1 and sys.argv[1] == 'y':
        print("Plotting enabled")
        plotting_enabled = True
    else:
        print("Plotting disabled")
        plotting_enabled = False
    os.remove('log.txt') if os.path.exists('log.txt') else None
    ga_instance.run()