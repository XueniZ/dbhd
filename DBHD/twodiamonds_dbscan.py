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

X, y = read_file('twodiamonds', 0)
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


'''
for sample in np.arange(0, 200, 1):
    samples.append(sample)
    dbscan = DBSCAN(eps = 0.08, min_samples = sample).fit(X_array)
    y_pred = dbscan.labels_
    results_ami.append(adjusted_mutual_info_score(y, y_pred))
    results_nmi.append(normalized_mutual_info_score(y, y_pred))

plt.plot(eps_value, results_ami, c = 'blue', label = 'AMI Score')
plt.plot(eps_value, results_nmi, c = 'red', label = 'NMI Score')
plt.xlabel('epsilon value')
plt.ylabel('Score Values')
plt.title("AMI and NMI results for different epsilon values(estimated n_5)")
plt.annotate(f'Max_AMI: {max_ami_results}', xy=(eps_value[max_index], max_ami_results), xytext=(10, -20), textcoords='offset points')
plt.annotate(f'Max_NMI: {max_nmi_results}', xy=(eps_value[max_nmi_index], max_nmi_results), xytext=(10, -50), textcoords='offset points')
plt.legend()
plt.show()


plt.plot(samples, results_ami, c = 'blue', label = 'AMI Score')
plt.plot(samples, results_nmi, c = 'red', label = 'NMI Score')
plt.xlabel('min samples')
plt.ylabel('Score Values')
plt.title("AMI and NMI results for different samples")
plt.annotate(f'Max_AMI: {max_ami_results}', xy=(samples[max_index], max_ami_results), xytext=(10, -20), textcoords='offset points')
plt.annotate(f'Max_NMI: {max_nmi_results}', xy=(samples[max_nmi_index], max_nmi_results), xytext=(10, -50), textcoords='offset points')
plt.legend()
plt.show()
'''