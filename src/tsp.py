import concurrent.futures
import sys
import numpy as np
import joblib
import queue
from bounding import *
from ML_feature import *

def tsp_ml(graph, bound=bound_bf, start_cost=sys.maxsize, init_visited=[0]):
    model = joblib.load('model/rf_classifier5-10.pkl')
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


        branch_prob = []
        for next_city in range(n):
            if next_city not in visited:
                new_cost = current_cost + graph[visited[-1]][next_city]
                features = extract_branch_features(graph, visited + [next_city], new_cost) if len(visited) < min(3, n-1) else None
                branch_prob.append((visited + [next_city], new_cost, model.predict_proba([features])[0][1] if len(visited) < min(3, n-1) else -new_cost))

        branch_prob.sort(key=lambda x: x[2], reverse=True)

        for new_visited, new_cost, prob in branch_prob:
            if bound(graph, new_visited, new_cost) < min_cost:
                branch(new_visited, new_cost)

    branch(init_visited, sum([graph[init_visited[i]][init_visited[i+1]] for i in range(len(init_visited) - 1)]) if len(init_visited) > 1 else 0)
    return min_cost, best_path, level_freq


def tsp(graph, bound=bound_bf, start_cost=sys.maxsize, init_visited=[0]):
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
                # print(min_cost)
            return

        for next_city in range(n):
            if next_city not in visited:
                # if len(visited)==1:
                #     tt=time.time()
                new_cost = current_cost + graph[visited[-1]][next_city]
                if bound(graph, visited + [next_city], new_cost) < min_cost:
                    branch(visited + [next_city], new_cost)
                # if len(visited)==1:
                #     print(': %s' % (time.time() - tt))


    branch(init_visited, sum([graph[init_visited[i]][init_visited[i+1]] for i in range(len(init_visited) - 1)]) if len(init_visited) > 1 else 0)
    return min_cost, best_path, level_freq

def tsp_process(graph, bound, init_visited, start_cost):
    # tt=time.time()
    n = len(graph)
    min_cost = start_cost
    best_path = []
    start = init_visited[0]
    # lock = threading.Lock()

    def branch(visited, current_cost):
        nonlocal min_cost, best_path

        if len(visited) == n:
            total_cost = current_cost + graph[visited[-1]][start]
            if total_cost < min_cost:
                min_cost = total_cost
                best_path[:] = visited[:] + [start]
                # print(min_cost)
            return

        for next_city in range(n):
            if next_city not in visited:
                new_cost = current_cost + graph[visited[-1]][next_city]
                if bound(graph, visited + [next_city], new_cost) < min_cost:
                    branch(visited + [next_city], new_cost)

    branch(init_visited, sum([graph[init_visited[i]][init_visited[i+1]] for i in range(len(init_visited) - 1)]) if len(init_visited) > 1 else 0)
    return min_cost, best_path
    # print(': %s' % (time.time() - tt))

def tsp_mp(graph, n_process, depth, bound=bound_bf, start_cost=sys.maxsize, start=0): 
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
            
            futures = [executor.submit(tsp_process, graph, bound, init_visited, min_cost) for init_visited in round_paths]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
            
            for result in results:
                if result[0] < min_cost:
                    min_cost = result[0]
                    best_path = result[1]

        def fill_init_path(path):
            if(len(path) == depth+1):
                init_path.put(path)
                if len(init_path.queue) == n_process:
                    one_parallel_round()
            else:
                for i in range(len(graph)):
                    if i not in path:
                        fill_init_path(path + [i])
        
        fill_init_path([start])
        if not init_path.empty():
            one_parallel_round()

    return min_cost, best_path
