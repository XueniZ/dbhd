from LDClusAlgo import LDClus
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
import arff
import umap
import math
import sys
from sklearn.preprocessing import LabelEncoder
from beta import estimated_beta
from beta import find_sigma
from beta import find_epsilon
from read_file import read_file

sys.setrecursionlimit(1000000)


X, y = read_file('segment', 1)
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

#rho = 1.8


results_ami = []
results_nmi = []
#index = 0
for n in cluster_size:
    for rho in rho_value:   

        sigma = find_sigma(X, n)
        eps = find_epsilon(X, n)
        esti_beta = estimated_beta(X, sigma, eps)
        print('beta' , esti_beta)

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
plt.plot(beta_value, results_ami, c = 'blue', label = 'AMI Score')
plt.plot(beta_value, results_nmi, c = 'red', label = 'NMI Score')
plt.title('AMI and NMI results with different beta values(estimated n_5)')
plt.xlabel('beta value(0-1)')
plt.ylabel('Score Values')
plt.annotate(f'Max_AMI: {max_ami_results}', xy=(beta_value[max_ami_index], max_ami_results), xytext=(10, -20), textcoords='offset points')
plt.annotate(f'Max_NMI: {max_nmi_results}', xy=(beta_value[max_nmi_index], max_nmi_results), xytext=(10, -70), textcoords='offset points')
plt.legend()
plt.show()
'''
'''
beta_variance = np.var(results)
beta_stand_deviation = np.std(results)

plt.scatter(beta_value, results)
plt.xlabel('beta')
plt.ylabel('AMI results')
plt.title("AMI results for n = log(X.size^2) in different beta values(distance: 0.01)")
plt.annotate(f'Variance: {beta_variance:.3f}', xy=(1.02, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
plt.annotate(f'Standard Deviation: {beta_stand_deviation:.3f}', xy=(1.05, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
plt.show()
'''