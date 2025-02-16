import numpy as np
from bounding import *
from data_gen import *
from tsp import *

def extract_branch_features(graph, visited, current_cost):
    """Extracts features for ML-based branch prioritization."""
    n = len(graph)
    unvisited = {i for i in range(n) if i not in visited}

    # MST cost of remaining nodes
    mst_cost = prim_mst(graph, unvisited)
    mst_cost_visited = prim_mst(graph, visited)
    mst_cost_all = prim_mst(graph, range(n))
    
    # Find two smallest edges connecting visited and unvisited nodes
    start_edges = []
    end_edges = []
    for u in unvisited:
        start_edges.append(graph[visited[-1]][u])
        end_edges.append(graph[visited[0]][u])
    

    return [
        current_cost,  # Current path cost
        len(unvisited),  # Remaining unvisited cities
        len(visited),  # Visited cities
        mst_cost,  # MST cost of remaining nodes
        min(start_edges),
        max(start_edges),
        np.mean(start_edges),
        np.std(start_edges),
        min(end_edges),
        max(end_edges),
        np.mean(end_edges),
        np.std(end_edges),
        graph[visited[-1]][visited[0]]
    ]