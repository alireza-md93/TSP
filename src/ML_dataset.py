import pickle
import numpy as np
from tqdm import tqdm
import sys
import concurrent.futures
import data_gen
import tsp
import ML_feature

# Generate data for ML-based branch prioritization
# takes a graph and generates random partial tours
def generate_data_with_one_graph(len_graph, n_branching):
    # Simulate branching decisions
    graph = data_gen.graph_gen(len_graph)
    for _ in range(n_branching):  # Simulate 10 branching choices per graph
        visited = list(np.random.choice(len_graph, size=np.random.randint(1, len_graph - 1), replace=False))
        current_cost = np.sum([graph[visited[i]][visited[i+1]] for i in range(len(visited)-1)]) if len(visited) > 1 else 0
        features = ML_feature.extract_branch_features(graph, visited, current_cost)
        optimal_tour_cost, _, _ = tsp.tsp(graph, bound=tsp.tsp_utility.bound_edge, start_cost=tsp.tsp_utility.cost_estimate(graph))
        best_achievable_cost, _, _ = tsp.tsp(graph, bound=tsp.tsp_utility.bound_edge, init_visited=visited)
        remaining_cost = best_achievable_cost - current_cost
        # Label: 1 if the branch led to an optimal or near-optimal solution, else 0
        label = 1 if best_achievable_cost <= optimal_tour_cost * 1.2 else 0  # Allow 20% deviation

        return features, label, remaining_cost

#
# Generate data for ML-based branch prioritization
# generates random graphs and calls generate_data_with_one_graph to apply different partial tours
# len_graph: list of graph sizes
# n_branching: number of branching decisions per graph
# N: number of graphs to generate
# parallel: number of parallel processes
# filename: file to save the generated data
def genrate_data_and_save(len_graph, n_branching, N, parallel, filename):
    X_train, Y1_train, Y2_train = [], [], []
    for l in len_graph:
        for i in tqdm (range (N//parallel), desc="Data Generation for L=" + str(l) + "..."):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(generate_data_with_one_graph, l, n_branching) for i in range(parallel)]
                for future in concurrent.futures.as_completed(futures):
                    x, y1, y2 = future.result()
                    X_train.append(x)
                    Y1_train.append(y1)
                    Y2_train.append(y2)
        
    with open(filename, 'wb') as f:
        pickle.dump((X_train, Y1_train, Y2_train), f)

def dummy(n):
    return n
    
if __name__ == "__main__":
    genrate_data_and_save(len_graph=range(5,11), n_branching=10, N=1000, parallel=32, filename='data/data.pkl')




    
    


