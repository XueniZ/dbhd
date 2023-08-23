from LDClusAlgo import LDClus
from LDClusAlgo import search_epsilon
#from LDClusAlgo import sigma_suche
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
import arff
from beta import estimated_beta
from beta import find_sigma
from beta import find_epsilon
import sys

sys.setrecursionlimit(1000000)

with open('/Users/xueni/Desktop/DBHD/data/twenty.arff', 'r') as f:
    arff_data = arff.load(f)


data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])

data['class'] = data['class'].astype(int)

X = data.iloc[:, :-1]  # Features (attributes)
y = data.iloc[:, -1]   # Class labels
X_array = X.values

rho = 1.8
#beta = 0.0
results_ami = []
results_nmi = []
cluster_size = []

for n in np.arange(5, 200, 1):
    cluster_size.append(n)

    sigma = find_sigma(X_array, n)
    eps = find_epsilon(X_array, n)
    esti_beta = estimated_beta(X_array, sigma, eps, n)

    y_pred = LDClus(X_array, n, rho, esti_beta) #return Labels
    #print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
    results_ami.append(adjusted_mutual_info_score(y, y_pred))
    results_nmi.append(normalized_mutual_info_score(y, y_pred))


n_variance = np.var(results_ami)
n_stand_deviation = np.std(results_ami)
#print(beta_variance)
#print(beta_stand_deviation)
#print(results)

max_ami_results = max(results_ami)
max_ami_index = results_ami.index(max_ami_results)

max_nmi_results = max(results_nmi)
max_nmi_index = results_nmi.index(max_nmi_results)

#plt.scatter(cluster_size, results_ami)
plt.plot(cluster_size, results_ami, c = 'blue', label = 'AMI Score')
plt.plot(cluster_size, results_nmi, c = 'red', label = 'NMI Score')
plt.title('AMI and NMI results with different cluster sizes(estimated beta_2)')
plt.xlabel('cluster size(5 to 200)')
plt.ylabel('Score Values')
#plt.annotate(f'Variance: {n_variance:.3f}', xy=(1.02, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
#plt.annotate(f'Standard Deviation: {n_stand_deviation:.3f}', xy=(1.05, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
plt.annotate(f'Max_AMI: {max_ami_results}', xy=(cluster_size[max_ami_index], max_ami_results), xytext=(10, -20), textcoords='offset points')
plt.annotate(f'Max_NMI: {max_nmi_results}', xy=(cluster_size[max_nmi_index], max_nmi_results), xytext=(10, -50), textcoords='offset points')
plt.legend()
plt.show()
