import time
from tqdm import tqdm
from data_gen import *
from tsp import *
from bounding import *

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
    validation_result = 'Success!'
    for i in tqdm (range (num), desc="Validation..."):
        graph = graph_gen(len_graph)
        best_cost_bf, best_path_bf, _ = tsp(graph)
        best_cost_edge, best_path_edge, _ = tsp(graph,bound_edge)
        best_cost_mst, best_path_mst, _ = tsp(graph,bound_mst)
        best_cost_edge_o, best_path_edge_o, _ = tsp(graph,bound_edge, cost_estimate(graph))
        best_cost_mst_o, best_path_mst_o, _ = tsp(graph,bound_mst, cost_estimate(graph))
        best_cost_edge_mp, best_path_edge_mp = tsp_mp(graph, 8, 3, bound_edge)
        best_cost_mst_mp, best_path_mst_mp = tsp_mp(graph, 8, 3, bound_mst)
        best_cost_edge_o_mp, best_path_edge_o_mp = tsp_mp(graph, 8, 3, bound_edge, cost_estimate(graph))
        best_cost_mst_o_mp, best_path_mst_o_mp = tsp_mp(graph, 8, 3, bound_mst, cost_estimate(graph))
        best_cost_ml_bf, best_path_ml_bf, _ = tsp_ml(graph)

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

        if  (best_path_bf == best_path_edge == best_path_mst == best_path_edge_o == best_path_mst_o) \
        and (best_cost_bf == best_cost_edge == best_cost_mst == best_cost_edge_o == best_cost_mst_o) \
        and (best_cost_bf == best_cost_edge_mp == best_cost_mst_mp == best_cost_edge_o_mp == best_cost_mst_o_mp) \
        and (best_cost_bf == best_cost_ml_bf) \
        and (check_path_equality(best_path_bf, best_path_edge_mp)   or check_cost_equality(graph, best_path_bf, best_path_edge_mp)   ) \
        and (check_path_equality(best_path_bf, best_path_mst_mp)    or check_cost_equality(graph, best_path_bf, best_path_mst_mp)    ) \
        and (check_path_equality(best_path_bf, best_path_edge_o_mp) or check_cost_equality(graph, best_path_bf, best_path_edge_o_mp) ) \
        and (check_path_equality(best_path_bf, best_path_mst_o_mp)  or check_cost_equality(graph, best_path_bf, best_path_mst_o_mp)  ) \
        and (check_path_equality(best_path_bf, best_path_ml_bf)     or check_cost_equality(graph, best_path_bf, best_path_ml_bf)     ):
            continue
        else:
            print(i)
            print('bf:        ' + str(best_path_bf)        + ' -> ' + str(best_cost_bf))
            print('edge:      ' + str(best_path_edge)      + ' -> ' + str(best_cost_edge))
            print('mst:       ' + str(best_path_mst)       + ' -> ' + str(best_cost_mst))
            print('edge_o:    ' + str(best_path_edge_o)    + ' -> ' + str(best_cost_edge_o))
            print('mst_o:     ' + str(best_path_mst_o)     + ' -> ' + str(best_cost_mst_o))
            print('edge_mp:   ' + str(best_path_edge_mp)   + ' -> ' + str(best_cost_edge_mp))
            print('mst_mp:    ' + str(best_path_mst_mp)    + ' -> ' + str(best_cost_mst_mp))
            print('edge_o_mp: ' + str(best_path_edge_o_mp) + ' -> ' + str(best_cost_edge_o_mp))
            print('mst_o_mp:  ' + str(best_path_mst_o_mp)  + ' -> ' + str(best_cost_mst_o_mp))
            print('ml_bf:     ' + str(best_path_ml_bf)     + ' -> ' + str(best_cost_ml_bf))
            validation_result = 'Failure!'
            break

    return validation_result

import joblib
import pickle
from sklearn.metrics import accuracy_score
if __name__ == "__main__":
    # with open('data/data5.pkl', 'rb') as f:
    #     x5,y5,d = pickle.load(f)
    # with open('data/data.pkl', 'rb') as f:
    #     x10,y10,d = pickle.load(f)
    # with open('model/rf_classifier5.pkl', 'rb') as f:
    #     model5 = joblib.load(f)
    # with open('model/rf_classifier.pkl', 'rb') as f:
    #     model10 = joblib.load(f)
    # with open('model/rf_classifier5-10.pkl', 'rb') as f:
    #     model510 = joblib.load(f)
    # y5_pred = model510.predict(x5)
    # y10_pred = model510.predict(x10)
    # print(accuracy_score(y5, y5_pred))
    # print(accuracy_score(y10, y10_pred))
    
    # validation_result = validation(10, 6)
    # print(validation_result)



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
