from sklearn.cluster import OPTICS
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

X, y = read_file('heart-statlog', 1)
from sklearn.preprocessing import MinMaxScaler
m = MinMaxScaler()
X = m.fit_transform(X)
label_encoder = LabelEncoder()
labels_numeric = label_encoder.fit_transform(y)

reducer = umap.UMAP(n_components=2, random_state = 42)
embedding = reducer.fit_transform(X)

results_ami = []
results_nmi = []
samples = np.arange(5, 200, 5)
xi_values = np.arange(0.01, 1, 0.1)

for sample in samples:
    for xi in xi_values:
        optics = OPTICS(min_samples = sample, xi = xi)
        y_pred = optics.fit_predict(X)
        results_ami.append(adjusted_mutual_info_score(y, y_pred))
        results_nmi.append(normalized_mutual_info_score(y, y_pred))


max_ami_results = max(results_ami)
max_index = results_ami.index(max_ami_results)

max_nmi_results = max(results_nmi)
max_nmi_index = results_nmi.index(max_nmi_results)

print(max_ami_results)
print(max_nmi_results)