# Packing Optimization

This repository contains implementations of optimization algorithms for solving the problem of packing circles within a square. The goal is to maximize the radius of the circles while ensuring they do not overlap and remain within the square's boundaries.

## Algorithms Implemented

### 1. SLSQP (Sequential Least Squares Programming)
- **File**: [`SLSQP_Packing.py`](SLSQP_Packing.py)
- **Description**: Uses the `scipy.optimize.minimize` function to optimize the circle packing problem. Constraints ensure circles do not overlap and stay within the square.

### 2. PSO (Particle Swarm Optimization)
- **File**: [`PSO_Packing.py`](PSO_Packing.py)
- **Description**: Implements a global-best particle swarm optimization algorithm using the `pyswarms` library to solve the packing problem.

### 3. Genetic Algorithm (GA)
- **File**: [`genetic_packing.py`](genetic_packing.py)
- **Description**: Implements a genetic algorithm using the `pygad` library to optimize the packing of circles within a square. The algorithm evolves solutions over multiple generations to maximize the shared radius of the circles while ensuring they do not overlap and stay within the square's boundaries.

## Results Visualization
All three algorithms generate visualizations of the optimal packing configuration:
- Circles are plotted within the square.
- Metrics such as radius, density, and distance are computed and displayed.

## Dependencies
- Python 3.x
- Libraries:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `pandas`
  - `pyswarms`
  - `pygad`

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/packing-optimization.git
   cd packing-optimization
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the scripts:
   - For SLSQP:
     ```bash
     python SLSQP_Packing.py
     ```
   - For PSO:
     ```bash
     python PSO_Packing.py
     ```
   - For Genetic Algorithm:
     ```bash
     python genetic_packing.py
     ```

## File Structure
- `SLSQP_Packing.py`: Implementation of the SLSQP algorithm.
- `PSO_Packing.py`: Implementation of the PSO algorithm.
- `genetic_packing.py`: Implementation of the Genetic Algorithm.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: Documentation for the repository.
