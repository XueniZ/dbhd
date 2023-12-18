import hdbscan
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arff
from sklearn.neighbors import NearestNeighbors
from read_file import read_file

X, y = read_file('cluto-t4-8k', 0)
from sklearn.preprocessing import MinMaxScaler
m = MinMaxScaler()
X = m.fit_transform(X)

results_ami = []
results_nmi = []
cluster_size = np.arange(5, 200, 5)
pts_values = np.arange(5, 200, 5)

for n in cluster_size:
    for pts in pts_values:
        n = int(n)
        pts = int(pts)
        hdb = hdbscan.HDBSCAN(min_cluster_size=n, min_samples=pts)
        y_pred = hdb.fit_predict(X)
        #print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
        results_ami.append(adjusted_mutual_info_score(y, y_pred))
        results_nmi.append(normalized_mutual_info_score(y, y_pred))

max_ami_results = max(results_ami)
max_index = results_ami.index(max_ami_results)

max_nmi_results = max(results_nmi)
max_nmi_index = results_nmi.index(max_nmi_results)

print(max_ami_results)
print(max_nmi_results)