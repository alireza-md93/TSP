import random
import numpy as np

def graph_gen(n_city):

    cityLoc = []
    while len(cityLoc) < n_city:
        newLoc = [random.randint(1, n_city), random.randint(1, n_city)]
        if not newLoc in cityLoc:
            cityLoc.append(newLoc)

    # for i in range(5):
    #     cityLoc[i][0] = cityLoc[i][0] + random.random()/0.5
    #     cityLoc[i][1] = cityLoc[i][1] + random.random()/0.5
    
    graph = np.array([[np.linalg.norm(np.array(cityLoc[i]) - np.array(cityLoc[j])) for j in range(n_city)] for i in range(n_city)])
    # graph = np.round(graph)
    # print(cityLoc)
    # print(graph)

    # Example Graph (Adjacency Matrix)
    # graph = [
    #     [0, 29, 20, 21],
    #     [29, 0, 15, 17],
    #     [20, 15, 0, 28],
    #     [21, 17, 28, 0]
    # ]

    return graph
