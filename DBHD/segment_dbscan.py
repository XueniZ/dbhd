from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arff
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import umap
from read_file import read_file

X, y = read_file('segment', 1)
from sklearn.preprocessing import MinMaxScaler
m = MinMaxScaler()
X = m.fit_transform(X)
label_encoder = LabelEncoder()
labels_numeric = label_encoder.fit_transform(y)

reducer = umap.UMAP(n_components=2, random_state = 42)
embedding = reducer.fit_transform(X)


results_ami = []
results_nmi = []
eps_value = np.arange(0.01, 1.01, 0.01)
samples = np.arange(5, 200, 5)

'''
for eps in np.arange(0.1, 1, 0.01):
    eps_value.append(eps)
    dbscan = DBSCAN(eps, min_samples=500).fit(features_array)
    y_pred = dbscan.labels_
    #print(y_pred)
    results_ami.append(adjusted_mutual_info_score(class_label, y_pred))
    results_nmi.append(normalized_mutual_info_score(class_label, y_pred))

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


