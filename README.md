# Travelling Salesman Problem (TSP)

## Overview
This project implements a solver for the Travelling Salesman Problem (TSP) with optimizations leveraging parallel computing and machine learning algorithms.

## Prerequisites
Ensure the following Python libraries are installed:
```bash
pip install numpy scikit-learn
```

## Machine Learning Models for Branch Prioritization
This implementation utilizes two machine learning models to prioritize branching decisions, reducing the number of branches needed for convergence.

### Generating Models
To generate datasets and train the models (Random Forest and Neural Network), execute the following commands:
```bash
python srs/ML_dataset.py  # Generates dataset and stores it in the data directory
python srs/ML_model.py    # Trains models and stores them in the data directory
```

### Dataset Generation
The function `generate_data_and_save` in `ML_dataset.py` runs in parallel. It generates random graphs, finds random partial paths, extracts features, and calculates the best possible solutions. The labeling strategy is as follows:
- **Random Forest:** Labeled `1` if the achievable cost is at most 20% higher than the optimal tour cost.
- **Neural Network:** Outputs the remaining cost estimate.

#### Parameters:
- `len_graph`: List of graph sizes.
- `n_branching`: Number of branching decisions per graph.
- `N`: Number of graphs to generate.
- `parallel`: Number of parallel processes.
- `filename`: Output file path for generated data.

### Model Training
The script `ML_model.py` generates and stores the models using default configurations. Modify this script to adjust hyperparameters or training settings.

## Running the Solver
The solver provides both single-process and multi-process implementations, with optional ML-based branch prioritization and various lower-bound calculation methods.

### Single-Process Solver (`tsp.tsp`)
#### Arguments:
- `graph`: Distance matrix.
- `bound`: Function to calculate the lower bound of cost. Options:
  - `tsp.tsp_utility.bound_bf` (Brute Force, default)
  - `tsp.tsp_utility.bound_edge` (Sum of minimum edges)
  - `tsp.tsp_utility.bound_mst` (MST heuristic)
- `prioritizer`: Function for branch prioritization. Options:
  - `tsp.tsp_utility.priority_none` (No priority, default)
  - `tsp.tsp_utility.priority_rf` (Random Forest)
  - `tsp.tsp_utility.priority_nn` (Neural Network)
- `model_path`: Path to the trained model file (default: `None`).
- `depth`: Number of cities considered for ML-based prioritization (default: `3`).
- `start_cost`: Initial cost estimate. Options:
  - `sys.maxsize` (default)
  - `tsp.tsp_utility.cost_estimate(graph)`
- `init_visited`: List of initially visited cities (default: `[0]`).

#### Returns:
- `min_cost`: Minimum path cost.
- `best_path`: Optimal path.
- `level_freq`: Frequency of each branching level.

### Multi-Process Solver (`tsp.tsp_mp`)
#### Arguments:
- `graph`: Distance matrix.
- `n_process`: Number of parallel processes.
- `depth_p`: Branch tree depth to allocate per process.
- `bound`: Lower bound calculation function (same options as `tsp.tsp`).
- `prioritizer`: Branch prioritization function (same options as `tsp.tsp`).
- `model_path`: Path to the trained model file (default: `None`).
- `depth`: Number of cities considered for ML-based prioritization (default: `3`).
- `start_cost`: Initial cost estimate (same options as `tsp.tsp`).
- `init_visited`: List of initially visited cities (default: `[0]`).

#### Returns:
- `min_cost`: Minimum path cost.
- `best_path`: Optimal path.

## Validation
To validate the project across all configurations, run:
```bash
python src/validation.py
```
This script runs all solvers with random graphs of size 6 for 10 iterations.

## Experiments

To run custom experiments, modify the `result.py` file in the specified section. The function `do_experiment` should be called as many times as needed. Results will be stored in `results.csv` and appended to existing results if the file already exists.

Each experiment is assigned a unique ID:

If `results.csv` does not exist, the first experiment starts with ID 0.
If results.csv exists, the ID continues from the last recorded experiment.

### Experiment Handler (`do_experiment`)
#### Arguments:
- `graph` : Adjacency matrix of the graph.
- `bound` : Method to calculate the lower bound of cost. Options:
  - `'brute force'`
  - `'edge'`
  - `'MST'`
- `prioritizer` : Method to prioritize branches. Options:
  - `'none'`
  - `'random forest'`
  - `'neural network'`
- `depth_ml` : Number of cities considered for the ML model.
- `depth_p` : Branch tree depth allocated per process.
- `n_p` : Number of processes (1 for a single-process run).
- `start_cost` : Initial cost estimate. Options:
  - `'inf'`
  - `'estimate'`

## Conclusion
This project demonstrates an optimized approach to solving TSP using parallel computing and machine learning models. Further improvements can be made by experimenting with different ML architectures, hyperparameters, and heuristics.
