import random
import numpy as np

#gnerate distace matrix for n_city cities
#distances are euclidean distances between random points in 2D plane
def graph_gen(n_city):

    cityLoc = []
    while len(cityLoc) < n_city:
        newLoc = [random.randint(1, n_city), random.randint(1, n_city)]
        if not newLoc in cityLoc:
            cityLoc.append(newLoc)

    # for i in range(5):
    #     cityLoc[i][0] = cityLoc[i][0] + random.random()/0.5
    #     cityLoc[i][1] = cityLoc[i][1] + random.random()/0.5
    
    # calculate euclidean distances
    graph = np.array([[np.linalg.norm(np.array(cityLoc[i]) - np.array(cityLoc[j])) for j in range(n_city)] for i in range(n_city)])

    return graph
