from LDClusAlgo import LDClus
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import arff
import math
import umap
from sklearn.preprocessing import LabelEncoder
from read_file import read_file
from beta import estimated_beta
from beta import find_sigma
from beta import find_epsilon

X, y = read_file('wine', 1)
from sklearn.preprocessing import MinMaxScaler
m = MinMaxScaler()
X = m.fit_transform(X)

label_encoder = LabelEncoder()
labels_numeric = label_encoder.fit_transform(y)

reducer = umap.UMAP(n_components=2, random_state = 42)
embedding = reducer.fit_transform(X)

cluster_size = np.arange(5, 200, 5)
rho_value = np.arange(0.05, 2.05, 0.05)
beta_value = np.arange(0, 0.7, 0.1)

results_ami = []
results_nmi = []

for n in cluster_size:
    for rho in rho_value:

        sigma = find_sigma(X, n)
        eps = find_epsilon(X, n)
        esti_beta = estimated_beta(X, sigma, eps)

        y_pred = LDClus(X, n, rho, esti_beta) #return Labels
        #print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
        results_ami.append(adjusted_mutual_info_score(y, y_pred))
        results_nmi.append(normalized_mutual_info_score(y, y_pred))

max_ami_results = max(results_ami)
max_ami_index = results_ami.index(max_ami_results)

max_nmi_results = max(results_nmi)
max_nmi_index = results_nmi.index(max_nmi_results)

print(max_ami_results)
print(max_nmi_results)
