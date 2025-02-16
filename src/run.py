import time
from tqdm import tqdm
from data_gen import *
from tsp import *
from tsp_utility import *

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
        graph = graph_gen(len_graph)
        best_cost_bf, best_path_bf, _ = tsp(graph)
        for bound in [bound_bf, bound_edge, bound_mst]:
            for prioritizer, model_path in [(priority_none, None), (priority_rf, 'model/rf_classifier.pkl'), (priority_nn, 'model/mlp_regressor.pkl')]:
                for start_cost in [sys.maxsize, cost_estimate(graph)]:
                    best_cost, best_path, _ = tsp(graph, bound=bound, prioritizer=prioritizer, model_path=model_path, depth=4, start_cost=start_cost, init_visited=[0])
                    if (best_cost_bf != best_cost) or not ( check_path_equality(best_path_bf, best_path) or check_cost_equality(graph, best_path_bf, best_path)):
                        print(i)
                        print('bound: ' + str(bound))
                        print('prioritizer: ' + str(prioritizer))
                        print('model_path: ' + str(model_path))
                        print('start_cost: ' + str(start_cost))
                        print('bf: ' + str(best_path_bf) + ' -> ' + str(best_cost_bf))
                        print('best: ' + str(best_path) + ' -> ' + str(best_cost))
                        return 'Failure!'
                    best_cost_mp, best_path_mp = tsp_mp(graph, n_process=8, depth_p=3, bound=bound, prioritizer=prioritizer, model_path=model_path, depth_ml=3, start_cost=start_cost, init_visited=[0])
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

    graph = graph_gen(14)
    
    # with Timer('brute_force'):
    #     best_cost_bf, best_path_bf = tsp(graph)

    # with Timer('tsp_branch_and_bound'):
    #     best_cost, best_path = tsp_branch_and_bound(graph)

    # with Timer('bf'):
    #     best_cost_edge, best_path_edge = tsp(graph)

    with Timer('edge'):
        best_cost_edge, best_path_edge, level_freq = tsp(graph, bound_edge)
    print(best_cost_edge)
    print(best_path_edge)
    print(level_freq)

    # with Timer('edge'):
    #     best_cost_edge, best_path_edge = tsp(graph,bound_edge, cost_estimate(graph))
    # print(best_cost_edge)
    # print(best_path_edge)

    with Timer('edge multi-threaded'):
        best_cost_edge_mt, best_path_edge_mt = tsp_mp(graph,4,4, bound_edge)
    print(best_cost_edge_mt)
    print(best_path_edge_mt)

    # with Timer('ml'):
    #     best_cost_edge_mt, best_path_edge_mt, level_freq = tsp_ml(graph,bound_edge)
    # print(best_cost_edge_mt)
    # print(best_path_edge_mt)
    # print(level_freq)

    # with Timer('mst'):
    #     best_cost_mst, best_path_mst = tsp(graph,bound_mst)


    # print("Optimal Cost:", best_cost)
    # print("Optimal Path:", best_path)
    # best_cost2, best_path2 = tsp(graph)
    # print("Optimal Cost:", best_cost2)
    # print("Optimal Path:", best_path2)
    # best_cost3, best_path3 = tsp(graph,bound_edge)
    # print("Optimal Cost:", best_cost3)
    # print("Optimal Path:", best_path3)
