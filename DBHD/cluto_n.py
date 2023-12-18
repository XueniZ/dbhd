from LDClusAlgo import LDClus
from LDClusAlgo import search_epsilon
#from LDClusAlgo import sigma_suche
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
import arff
import math
from beta import estimated_beta
from beta import find_sigma
from beta import find_epsilon
import sys
import faulthandler
from read_file import read_file

faulthandler.enable()
sys.setrecursionlimit(1000000)


X, y = read_file('cluto-t4-8k', 0)
from sklearn.preprocessing import MinMaxScaler
m = MinMaxScaler()
X = m.fit_transform(X)

cluster_size = np.arange(5, 200, 5)
rho_value = np.arange(0.05, 2.05, 0.05)
beta_value = np.arange(0, 0.7, 0.1)

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
estimate_n_4 = 2 * X.shape[1]

#fünfte Formel
minClusterSize_5 = math.log2(len(X)) + X.shape[1]
estimate_n_5 = round(minClusterSize_5)

'''
fig, axs = plt.subplots(5, 4, figsize=(12, 15))

for i in range(5):
    for j in range(4):
        rho = rho_value[i * 4 + j]
        ami_scores = []
        nmi_scores = []
        for beta in beta_value:
            y_pred = LDClus(X, estimate_n_1, rho, beta) #return Labels
        
            ami = adjusted_mutual_info_score(y, y_pred)
            nmi = normalized_mutual_info_score(y, y_pred)
        
            ami_scores.append(ami)
            nmi_scores.append(nmi)
    
        axs[i, j].plot(beta_value, ami_scores, label='AMI')
        axs[i, j].plot(beta_value, nmi_scores, label='NMI')
        axs[i, j].set_title(f'Rho = {rho: .1f}')
        axs[i, j].set_ylabel('Scores')
        axs[i, j].legend()

for ax in axs.flat:
    ax.label_outer()

plt.xlabel('Beta Values')
plt.tight_layout()
plt.show()
'''


#n = 95
#rho = 1.8
#beta = 0.0
results_ami = []
results_nmi = []
'''
for n in cluster_size:
    for rho in rho_value:
        for beta in beta_value:
            y_pred = LDClus(X, n, rho, beta) #return Labels
            #print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
            results_ami.append(adjusted_mutual_info_score(y, y_pred))
            results_nmi.append(normalized_mutual_info_score(y, y_pred))
    

max_ami_results = max(results_ami)
max_ami_index = results_ami.index(max_ami_results)

max_nmi_results = max(results_nmi)
max_nmi_index = results_nmi.index(max_nmi_results)

print(max_ami_results)
print(max_nmi_results)
'''

for rho in rho_value:
    for beta in beta_value:
        y_pred = LDClus(X, estimate_n_5, rho, beta) #return Labels
        #print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
        results_ami.append(adjusted_mutual_info_score(y, y_pred))
        results_nmi.append(normalized_mutual_info_score(y, y_pred))
    

max_ami_results = max(results_ami)
max_ami_index = results_ami.index(max_ami_results)

max_nmi_results = max(results_nmi)
max_nmi_index = results_nmi.index(max_nmi_results)

print(max_ami_results)
print(max_nmi_results)
