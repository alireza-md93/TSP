import time
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import data_gen
import tsp

exp_id = 0
results = []

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print()
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

# tic toc functions to measure the elapsed time
def tic():
    global start_time
    start_time = time.time()
def toc():
    elapsed_time = time.time() - start_time
    return elapsed_time

# append the result of the experiment as a dictionary to the results list    
def append_result(graph, path, cost, level_freq, time, bound='brute force', prioritizer='none', depth_ml=0, depth_p=1, start_cost='inf'):
    global exp_id
    mst_cost = tsp.tsp_utility.prim_mst(graph, range(len(graph)))
    edges = [graph[i][j] for i in range(len(graph)) for j in range(i+1, len(graph))]
    min_edge = np.min(edges)
    max_edge = np.max(edges)
    mean_edge = np.mean(edges)
    std_edge = np.std(edges)
    result = {
        'experiment id': exp_id,
        'best path': path,
        'best cost': cost,
        'node expansion': level_freq,
        'elapsed time': time,
        'graph size': len(graph),
        'graph MST cost': mst_cost,
        'estimated cost': tsp.tsp_utility.cost_estimate(graph),
        'graph min edge': min_edge,
        'graph max edge': max_edge,
        'graph mean edge': mean_edge,
        'graph std edge': std_edge,
        'bounding method': bound,
        'inital cost': start_cost,
        'ML prioritizer': prioritizer,
        'ML depth': depth_ml,
        'multi process depth': depth_p
    }
    results.append(result)
    exp_id += 1

# run the experiment with the given parameters and append the result to the results list
# graph: adjacency matrix of the graph
# bound: the method to calculate the lower bound of the cost ('brute force', 'edge', 'MST')
# prioritizer: the method to prioritize the branch ('none', 'random forest', 'neural network')
# depth_ml: the number of cities to consider for the ML model
# depth_p: the level of the branch tree to consider for each process
# n_p: number of processes (1 for single process)
# start_cost: initial cost of the path ('inf', 'estimate')
def do_experiment(graph, bound='brute force', prioritizer='none', depth_ml=0, depth_p=0, n_p=1, start_cost='inf'):
    bound_func = {'brute force' : tsp.tsp_utility.bound_bf, 'edge': tsp.tsp_utility.bound_edge, 'MST': tsp.tsp_utility.bound_mst}
    prioritizer_func = {'none': tsp.tsp_utility.priority_none, 'random forest': tsp.tsp_utility.priority_rf, 'neural network': tsp.tsp_utility.priority_nn}
    model_path = {'none' : None , 'random forest': 'model/rf_classifier.pkl', 'neural network': 'model/mlp_regressor.pkl'}
    start_cost_val = {'inf': sys.maxsize, 'estimate': tsp.tsp_utility.cost_estimate(graph)}

    if(n_p == 1):
        tic()
        best_cost, best_path, level_freq = tsp.tsp(
            graph, 
            bound=bound_func[bound], 
            prioritizer=prioritizer_func[prioritizer], 
            model_path=model_path[prioritizer], 
            depth=depth_ml, 
            start_cost=start_cost_val[start_cost], 
            init_visited=[0]
        )
        elapsed_time = toc()
    else:
        tic()
        best_cost, best_path = tsp.tsp_mp(
            graph, 
            n_process=n_p, 
            depth_p=depth_p,
            bound=bound_func[bound], 
            prioritizer=prioritizer_func[prioritizer], 
            model_path=model_path[prioritizer], 
            depth_ml=depth_ml, 
            start_cost=start_cost_val[start_cost], 
            init_visited=[0]
        )
        elapsed_time = toc()
        level_freq = None
    
    append_result(
        graph, 
        best_path, 
        best_cost, 
        level_freq, 
        elapsed_time, 
        bound=bound, 
        prioritizer=prioritizer, 
        depth_ml=depth_ml, 
        depth_p=depth_p, 
        start_cost=start_cost
    )

# edit the specified section to run your experiment
# the results will be saved in results.csv
# the results will be appended to the existing results if there is any
if __name__ == '__main__':
    # if results.csv exists, get the last experiment id
    try:
        old_df = pd.read_csv('results.csv')
        exp_id = old_df['experiment id'].max() + 1
    except FileNotFoundError:
        old_df = pd.DataFrame()
        exp_id = 0
     
    ####################### you experiment code here ########################
    for N in [8, 9, 10]:
        for i in tqdm(range(20), desc='Running single-process experiments for N='+str(N)+'...'):
            graph = data_gen.graph_gen(N)
            for bound_name in ['edge', 'MST']:
                for cost in ['inf', 'estimate']:
                    for prioritizer in ['none', 'random forest', 'neural network']:
                        for depth_ml in ([0] if prioritizer=='none' else [2]):
                            # do_experiment(graph, bound=bound_name, prioritizer=prioritizer, depth_ml=depth_ml, n_p=1, start_cost=cost)
                            for depth_p in [1, 2, 3]:
                                do_experiment(graph, bound=bound_name, prioritizer=prioritizer, depth_ml=depth_ml, n_p=8, depth_p=depth_p, start_cost=cost)
    ##########################################################################

    # save the results and append to the old results if exists
    df = pd.DataFrame(results) if old_df.empty else pd.concat([old_df, pd.DataFrame(results)], ignore_index=True)
    df.to_csv('results.csv', index=False)