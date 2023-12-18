from LDClusAlgo import LDClus
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import arff
import math
from beta import estimated_beta
from beta import find_sigma
from beta import find_epsilon
from read_file import read_file
import sys
import faulthandler


faulthandler.enable()
sys.setrecursionlimit(1000000)

X, y = read_file('twenty', 0)
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
minClusterSize_4 = 2 * X.shape[1]

#fünfte Formel
minClusterSize_5 = math.log2(len(X)) + X.shape[1]
estimate_n_5 = round(minClusterSize_5)

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

'''
fig, axs = plt.subplots(5, 4, figsize=(15,12))

for i in range(5):
    for j in range(4):
        rho = rho_value[i * 4 + j]
        ami_scores = []
        nmi_scores = []
        for n in cluster_size:
            sigma = find_sigma(X, n)
            eps = find_epsilon(X, n)
            esti_beta = estimated_beta(X, sigma, eps)
            #print('rho', rho, 'beta', esti_beta, 'n', n)
            y_pred = LDClus(X, n, rho, esti_beta) #return Labels
        
            ami = adjusted_mutual_info_score(y, y_pred)
            nmi = normalized_mutual_info_score(y, y_pred)
        
            ami_scores.append(ami)
            nmi_scores.append(nmi)
    
        axs[i, j].plot(cluster_size, ami_scores, label='AMI')
        axs[i, j].plot(cluster_size, nmi_scores, label='NMI')
        axs[i, j].set_title(f'Rho = {rho: .1f}')
        axs[i, j].set_ylabel('Scores')
        axs[i, j].legend()

for ax in axs.flat:
    ax.label_outer()

plt.xlabel('Cluster Size')
plt.tight_layout()
plt.show()
'''
#beta_variance = np.var(results)
#beta_stand_deviation = np.std(results)
'''
max_ami_results = max(results_ami)
max_index = results_ami.index(max_ami_results)

max_nmi_results = max(results_nmi)
max_nmi_index = results_nmi.index(max_nmi_results)

#plt.scatter(beta_value, results)
plt.plot(beta_value, results_ami, c = 'blue', label = 'AMI Score')
plt.plot(beta_value, results_nmi, c = 'red', label = 'NMI Score')
plt.title('AMI and NMI results for different cluster size and rho values with estimated beta2')
plt.xlabel('')
plt.ylabel('Score Values')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
#plt.annotate(f'Variance: {beta_variance:.3f}', xy=(1.02, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
#plt.annotate(f'Standard Deviation: {beta_stand_deviation:.3f}', xy=(1.05, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
#plt.annotate(f'Max_AMI: {max_ami_results}', xy=(beta_value[max_index], max_ami_results), xytext=(10, -20), textcoords='offset points')
#plt.annotate(f'Max_NMI: {max_nmi_results}', xy=(beta_value[max_nmi_index], max_nmi_results), xytext=(10, -70), textcoords='offset points')
plt.legend()
plt.show()
'''