
from LDClusAlgo import LDClus
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from LDClusAlgo import search_epsilon

#####beta estimate#####

def dist(x, y):
    #distance = math.sqrt(np.sum((x - y)**2))
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    #distance = np.sqrt(((x[0]-y[0])**2)+((x[1]-y[1])**2))
    distance = np.linalg.norm(x - y)
    #print('distance', distance)
    return distance

def eps_neighborhood(X, sigma, eps):
    #epsilon_neighbors = [x for x in X if dist(x, sigma) <= eps]
    #neighbors_array = np.array(epsilon_neighbors)
    epsilon_neighbors = []
    for x in X:
        D = np.array([point for point in X if not np.array_equal(point, x)])
        for x in D:
            if dist(x, sigma) <= eps:
                epsilon_neighbors.append(x)
    return epsilon_neighbors

def average_distance(X, sigma, eps):
    total_distance = 0.0
    avg_distance = 0.0
    eps_neighbors = eps_neighborhood(X, sigma, eps)
    for x in eps_neighbors:
        total_distance += dist(x, sigma)
    if len(eps_neighbors) != 0:
        avg_distance = total_distance / len(eps_neighbors)   
    return avg_distance

def n_nearest_neighbors(x, X, n):
    heap = []
    D = np.array([point for point in X if not np.array_equal(point, x)]) #the cluster with all unassigned data points
    distances = np.array([dist(x, point) for point in D])
    indices = np.argsort(distances)[:n]
    neighbors = D[indices]
    return neighbors

def avgQ(x, nbrs, n):
    #n_nearest = n_nearest_neighbors(x, X, n)

    sum_Q_values = sum(dist(x, nbr) for nbr in nbrs)
    return sum_Q_values / n

def find_sigma(X, n):
    avg_Q_values = []
    
    for x in X:
        # Filter out the data points that have already been assigned to a cluster
        D = np.array([point for point in X if not np.array_equal(point, x)]) #the cluster with all unassigned data points
        n_nearest = n_nearest_neighbors(x, D, n) #dataset, min cluster size
        avg_Q_values.append(avgQ(x, n_nearest, n))
    min_sigma_value = min(avg_Q_values)
    return min_sigma_value

def find_epsilon(X, n):
    max_distances = []
    
    for x in X:
        D = np.array([point for point in X if not np.array_equal(point, x)]) #the cluster with all unassigned data points
        n_nearest = n_nearest_neighbors(x, D, n)
        max_distance = max(dist(x, y) for y in n_nearest) # Calculate the maximum distance between x and its n-nearest neighbors
        max_distances.append(max_distance)
    min_epsilon_value = min(max_distances)
    return min_epsilon_value

def estimated_beta(X, sigma, eps):
    estimated_beta = 0.0
    results = 0.0
    std = 0.0
    var = 0.0
    print('start')
    eps_neighbors = eps_neighborhood(X, sigma, eps)
    #sigmas = [sigma for sigma in eps_neighborhood(X, sigma, eps)]
    print('length', len(eps_neighbors))
    for x in eps_neighbors:  
        avg_sigma = average_distance(x, sigma, eps)
        estimated_beta += (dist(x, sigma) - avg_sigma)**2
    print('estimate beta', estimated_beta)
    if len(eps_neighbors) != 0:

        var = estimated_beta / len(eps_neighbors) #first estimation
        std = var**0.5 #second estimation
        results = var * (1 + std) #third estimation
    return std

