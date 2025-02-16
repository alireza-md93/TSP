import concurrent.futures
import sys
import numpy as np
import joblib
import queue
import tsp_utility


############################################ Single Process ############################################
#### Arguments ####
# graph: distance matrix
# bound: function to calculate the lower bound of the cost (default: tsp_utility.bound_bf)
# prioritizer: function to prioritize the branch (default: tsp_utility.priority_none)
# model_path: path to the model file (default: None)
# depth: The number of cities to consider for ML model (default: 3)
# start_cost: initial cost of the path (default: sys.maxsize)
# init_visited: list of initial visited cities (default: [0])

#### Return ####
# min_cost: minimum cost of the path
# best_path: the best path
# level_freq: the frequency of each level in the branch tree

def tsp(graph, bound=tsp_utility.bound_bf, prioritizer=tsp_utility.priority_none, model_path=None, depth=3, start_cost=sys.maxsize, init_visited=[0]):
    model = joblib.load(model_path) if model_path != None else None
    n = len(graph)
    min_cost = start_cost
    best_path = []
    start = init_visited[0]
    level_freq = [0] * n

    def branch(visited, current_cost):
        nonlocal min_cost, best_path
        level_freq[len(visited)-1] += 1

        if len(visited) == n:
            total_cost = current_cost + graph[visited[-1]][start]
            if total_cost < min_cost:
                min_cost = total_cost
                best_path[:] = visited[:] + [start]
            return

        branch_priority = prioritizer(graph, visited, current_cost, model, depth)
        
        for new_visited, new_cost, prob in branch_priority:
            if bound(graph, new_visited, new_cost) < min_cost:
                branch(new_visited, new_cost)

    branch(init_visited, sum([graph[init_visited[i]][init_visited[i+1]] for i in range(len(init_visited) - 1)]) if len(init_visited) > 1 else 0)
    return min_cost, best_path, level_freq

############################################ Multi Process ############################################
# each process's task
def tsp_process(graph, bound, prioritizer, model, depth, start_cost, init_visited):
    n = len(graph)
    min_cost = start_cost
    best_path = []
    start = init_visited[0]

    def branch(visited, current_cost):
        nonlocal min_cost, best_path

        if len(visited) == n:
            total_cost = current_cost + graph[visited[-1]][start]
            if total_cost < min_cost:
                min_cost = total_cost
                best_path[:] = visited[:] + [start]
            return
        
        branch_priority = prioritizer(graph, visited, current_cost, model, depth)
        
        for new_visited, new_cost, prob in branch_priority:
            if bound(graph, new_visited, new_cost) < min_cost:
                branch(new_visited, new_cost)

    branch(init_visited, sum([graph[init_visited[i]][init_visited[i+1]] for i in range(len(init_visited) - 1)]) if len(init_visited) > 1 else 0)
    return min_cost, best_path

#### Arguments ####
# graph: distance matrix
# n_process: number of processes
# depth_p: The level of the branch tree to consider for each process
# bound: function to calculate the lower bound of the cost (default: tsp_utility.bound_bf)
# prioritizer: function to prioritize the branch (default: tsp_utility.priority_none)
# model_path: path to the model file (default: None)
# depth: The number of cities to consider for ML model (default: 3)
# start_cost: initial cost of the path (default: sys.maxsize)
# init_visited: list of initial visited cities (default: [0])

#### Return ####
# min_cost: minimum cost of the path
# best_path: the best path
# level_freq: the frequency of each level in the branch tree

# main function for multi-process
def tsp_mp(graph, n_process, depth_p, bound=tsp_utility.bound_bf, prioritizer=tsp_utility.priority_none, model_path=None, depth_ml=3, start_cost=sys.maxsize, init_visited=[0]): 
    model = joblib.load(model_path) if model_path != None else None
    init_path = queue.Queue()
    min_cost = start_cost
    best_path = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        def one_parallel_round():
            nonlocal min_cost, best_path
            round_paths = []
            results = []
            while not init_path.empty():
                round_paths += [init_path.get()]
            
            futures = [executor.submit(tsp_process, graph, bound, prioritizer, model, depth_ml, min_cost, visited) for visited in round_paths]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
            
            for result in results:
                if result[0] < min_cost:
                    min_cost = result[0]
                    best_path = result[1]

        def conut_paths(path, cost):
            if(len(path) == depth_p+1):
                init_path.put(path)
                if len(init_path.queue) == n_process:
                    one_parallel_round()
            else:
                branch_priority = tsp_utility.priority_none(graph, path, cost, model, depth_ml)
                for new_path, new_cost, prob in branch_priority:
                    conut_paths(new_path, new_cost)
        
        conut_paths(init_visited, sum([graph[init_visited[i]][init_visited[i+1]] for i in range(len(init_visited) - 1)]) if len(init_visited) > 1 else 0)
        if not init_path.empty():
            one_parallel_round()

    return min_cost, best_path
