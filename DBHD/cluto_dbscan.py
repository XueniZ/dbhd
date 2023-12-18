from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arff
from sklearn.neighbors import NearestNeighbors
import math
from read_file import read_file

X, y = read_file('cluto-t4-8k', 0)
from sklearn.preprocessing import MinMaxScaler
m = MinMaxScaler()
X = m.fit_transform(X)

results_ami = []
results_nmi = []
eps_value = np.arange(0.01, 1.01, 0.01)
samples = np.arange(5, 200, 5)

#erste Formel
minClusterSize_1 = math.log2(len(X)) + 5
#print(minClusterSize_1) 
estimate_n_1 = round(minClusterSize_1)

#zweite Formel
minClusterSize_2 = math.log2(len(X)^2)
estimate_n_2 = round(minClusterSize_2)


#dritte Formel
minClusterSize_3 = math.log2(len(X))
estimate_n_3 = round(minClusterSize_3)

#vierte Formel
minClusterSize_4 = 2 * X.shape[1]

#fünfte Formel
minClusterSize_5 = math.log2(len(X)) + X.shape[1]
estimate_n_5 = round(minClusterSize_5)


'''
####optimale epsilon bestimmen####
neighbors = NearestNeighbors(n_neighbors=50)
neighbors_fit = neighbors.fit(X_array)
distances, indices = neighbors_fit.kneighbors(X_array)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
#print(distances)
'''

for sample in samples:
    for eps in eps_value:
        dbscan = DBSCAN(eps=eps, min_samples=sample)
        y_pred = dbscan.fit_predict(X)
        #print(y_pred)
        results_ami.append(adjusted_mutual_info_score(y, y_pred))
        results_nmi.append(normalized_mutual_info_score(y, y_pred))


max_ami_results = max(results_ami)
max_ami_index = results_ami.index(max_ami_results)

max_nmi_results = max(results_nmi)
max_nmi_index = results_nmi.index(max_nmi_results)

print(max_ami_results)
print(max_nmi_results)
