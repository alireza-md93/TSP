import time
import sys
from tqdm import tqdm
import data_gen
import tsp

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

# Validation of all the configurations implemented in the project
# num: number of iterations
# len_graph: number of cities
# return: 'Success!' if all the configurations are implemented correctly, 'Failure!' otherwise
def validation(num, len_graph):
    def check_path_equality(p1, p2):
        p3 = p2.copy()
        p3.reverse()
        return p1 == p2 or p1 == p3
    
    def check_cost_equality(graph, p1, p2):
        cost1 = 0
        cost2 = 0
        for i in range(len(p1)-1):
            cost1 += graph[p1[i]][p1[i+1]]
            cost2 += graph[p2[i]][p2[i+1]]
        return cost1 == cost2
    
    for i in tqdm (range (num), desc="Validation..."):
        graph = data_gen.graph_gen(len_graph)
        best_cost_bf, best_path_bf, _ = tsp.tsp(graph)
        for bound in [tsp.tsp_utility.bound_bf, tsp.tsp_utility.bound_edge, tsp.tsp_utility.bound_mst]:
            for prioritizer, model_path in [(tsp.tsp_utility.priority_none, None), (tsp.tsp_utility.priority_rf, 'model/rf_classifier.pkl'), (tsp.tsp_utility.priority_nn, 'model/mlp_regressor.pkl')]:
                for start_cost in [sys.maxsize, tsp.tsp_utility.cost_estimate(graph)]:
                    best_cost, best_path, _ = tsp.tsp(graph, bound=bound, prioritizer=prioritizer, model_path=model_path, depth=4, start_cost=start_cost, init_visited=[0])
                    if (best_cost_bf != best_cost) or not ( check_path_equality(best_path_bf, best_path) or check_cost_equality(graph, best_path_bf, best_path)):
                        print(i)
                        print('bound: ' + str(bound))
                        print('prioritizer: ' + str(prioritizer))
                        print('model_path: ' + str(model_path))
                        print('start_cost: ' + str(start_cost))
                        print('bf: ' + str(best_path_bf) + ' -> ' + str(best_cost_bf))
                        print('best: ' + str(best_path) + ' -> ' + str(best_cost))
                        return 'Failure!'
                    best_cost_mp, best_path_mp = tsp.tsp_mp(graph, n_process=8, depth_p=3, bound=bound, prioritizer=prioritizer, model_path=model_path, depth_ml=3, start_cost=start_cost, init_visited=[0])
                    if (best_cost_bf != best_cost_mp) or not (check_path_equality(best_path_bf, best_path_mp) or check_cost_equality(graph, best_path_bf, best_path_mp)):
                        print(i)
                        print('bound: ' + str(bound))
                        print('prioritizer: ' + str(prioritizer))
                        print('model_path: ' + str(model_path))
                        print('start_cost: ' + str(start_cost))
                        print('bf: ' + str(best_path_bf) + ' -> ' + str(best_cost_bf))
                        print('best: ' + str(best_path_mp) + ' -> ' + str(best_cost_mp))
                        return 'Failure!'

    return 'Success!'

if __name__ == "__main__":  
    validation_result = validation(10, 6)
    print(validation_result)

