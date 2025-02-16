import pickle
import numpy as np
from tqdm import tqdm
import sys
import concurrent.futures
import multiprocessing.queues
from tsp_utility import *
from data_gen import *
from tsp import *
from ML_feature import *

def generate_data_with_one_graph(len_graph, n_branching):
    # Simulate branching decisions
    graph = graph_gen(len_graph)
    for _ in range(n_branching):  # Simulate 10 branching choices per graph
        visited = list(np.random.choice(len_graph, size=np.random.randint(1, len_graph - 1), replace=False))
        current_cost = np.sum([graph[visited[i]][visited[i+1]] for i in range(len(visited)-1)]) if len(visited) > 1 else 0
        features = extract_branch_features(graph, visited, current_cost)
        optimal_tour_cost, path1 = tsp(graph, bound_edge, cost_estimate(graph))
        best_achievable_cost, path2 = tsp(graph, bound_edge, sys.maxsize, visited)
        remaining_cost = best_achievable_cost - current_cost
        # Label: 1 if the branch led to an optimal or near-optimal solution, else 0
        label = 1 if best_achievable_cost <= optimal_tour_cost * 1.2 else 0  # Allow 20% deviation
        # print(graph)
        # print(visited)
        # print(current_cost)
        # print(features)
        # print(optimal_tour_cost)
        # print(best_achievable_cost)
        # print(path1)
        # print(path2)
        # print(label)

        return features, label, remaining_cost
        # X_train.append(features)
        # Y1_train.append(label)
        # Y2_train.append(best_achievable_cost - current_cost)

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




    
    


