import sys
import heapq
import ML_feature

def priority_rf(graph, visited, current_cost, model, depth):
    n = len(graph)
    branch_priority = []
    for next_city in range(n):
        if next_city not in visited:
            new_cost = current_cost + graph[visited[-1]][next_city]
            features = ML_feature.extract_branch_features(graph, visited + [next_city], new_cost) if len(visited) < min(n-1, depth) else None
            branch_priority.append((visited + [next_city], new_cost, model.predict_proba([features])[0][1] if len(visited) < min(n-1, depth) else -new_cost))

    branch_priority.sort(key=lambda x: x[2], reverse=True)
    return branch_priority

def priority_nn(graph, visited, current_cost, model, depth):
    n = len(graph)
    branch_priority = []
    for next_city in range(n):
        if next_city not in visited:
            new_cost = current_cost + graph[visited[-1]][next_city]
            features = ML_feature.extract_branch_features(graph, visited + [next_city], new_cost) if len(visited) < min(n-1, depth) else None
            branch_priority.append((visited + [next_city], new_cost, -model.predict([features])[0] if len(visited) < min(n-1, depth) else -new_cost))

    branch_priority.sort(key=lambda x: x[2], reverse=True)
    return branch_priority

def priority_none(graph, visited, current_cost, model, depth):
    n = len(graph)
    branch_priority = []
    for next_city in range(n):
        if next_city not in visited:
            new_cost = current_cost + graph[visited[-1]][next_city]
            branch_priority.append((visited + [next_city], new_cost, None))

    return branch_priority


def bound_edge(graph, visited, current_cost):
    n = len(graph)
    remaining_cost = sum(min(graph[i][j] for j in range(n) if i != j) for i in range(n) if i not in visited)
    return current_cost + remaining_cost
def cost_estimate(graph, start=0):
    n = len(graph)
    dist = [sys.maxsize for i in range(n)]
    visited = [start]
    unvisited = [i for i in range(n) if i not in visited]
    cost = 0

    while len(unvisited) > 0:
        for neighbour in unvisited:
            dist[neighbour] = graph[visited[-1]][neighbour]
        min_ind = dist.index(min(dist))
        cost = cost + dist[min_ind]
        visited.append(min_ind)
        unvisited.remove(min_ind)
        dist[min_ind] = sys.maxsize

    return (cost + graph[visited[-1]][start]) * 1.01

        
def prim_mst(graph, unvisited):
    """ Computes the Minimum Spanning Tree (MST) using Prim’s Algorithm. """
    if len(unvisited) < 2:
        return 0  # No MST needed for a single node

    min_cost = 0
    visited = set()
    start = next(iter(unvisited))  # Pick an arbitrary start node
    min_heap = [(0, start)]

    while min_heap and len(visited) < len(unvisited):
        cost, node = heapq.heappop(min_heap)
        if node in visited:
            continue
        visited.add(node)
        min_cost += cost

        for neighbor in unvisited:
            if neighbor not in visited:
                heapq.heappush(min_heap, (graph[node][neighbor], neighbor))

    return min_cost

def bound_mst(graph, visited, current_cost):
    """ Computes a lower bound using MST + Minimum Edge Heuristic """
    n = len(graph)
    unvisited = {i for i in range(n) if i not in visited}

    if not unvisited:
        return current_cost

    # Compute MST of unvisited cities
    mst_cost = prim_mst(graph, unvisited)

    # Find two smallest edges connecting visited and unvisited cities
    min_edges = []
    for v in visited:
        for u in unvisited:
            min_edges.append(graph[v][u])
    min_edges.sort()
    min_two_sum = sum(min_edges[:2]) if len(min_edges) >= 2 else sum(min_edges)

    return current_cost + 0.5 * (mst_cost + min_two_sum)

def bound_bf(a,b,c):
    return True
