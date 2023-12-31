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
import sys

sys.setrecursionlimit(1000000)

with open('/Users/xueni/Desktop/DBHD/data/twenty.arff', 'r') as f:
    arff_data = arff.load(f)


data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])

data['class'] = data['class'].astype(int)

X = data.iloc[:, :-1]  # Features (attributes)
y = data.iloc[:, -1]   # Class labels
X_array = X.values

n = 95
beta = 0.0
results_ami = []
results_nmi = []
rho_value = []

#erste Formel
minClusterSize_1 = math.log2(len(X_array)) + 5
#print(minClusterSize_1) 
estimate_n_1 = round(minClusterSize_1)

#zweite Formel
minClusterSize_2 = math.log2(len(X_array)^2)
estimate_n_2 = round(minClusterSize_2)

#dritte Formel
minClusterSize_3 = math.log2(len(X_array))
estimate_n_3 = round(minClusterSize_3)

#vierte Formel
minClusterSize_4 = 2 * X.shape[1]

#fünfte Formel
minClusterSize_5 = math.log2(len(X_array)) + X.shape[1]
estimate_n_5 = round(minClusterSize_5)

sigma = find_sigma(X_array, n)
eps = find_epsilon(X_array, n)
esti_beta = estimated_beta(X_array, sigma, eps, n)


for rho in np.arange(0.1, 2, 0.1):
    rho_value.append(rho)
    
    y_pred = LDClus(X_array, n, rho, esti_beta) #return Labels
    #print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
    results_ami.append(adjusted_mutual_info_score(y, y_pred))
    results_nmi.append(normalized_mutual_info_score(y, y_pred))

#rho_variance = np.var(results)
#rho_stand_deviation = np.std(results)

max_ami_results = max(results_ami)
max_index = results_ami.index(max_ami_results)

max_nmi_results = max(results_nmi)
max_nmi_index = results_nmi.index(max_nmi_results)

#plt.scatter(rho_value, results)
plt.plot(rho_value, results_ami, c = 'blue', label = 'AMI Score')
plt.plot(rho_value, results_nmi, c = 'red', label = 'NMI Score')
plt.xlabel('rho(0 to 2)')
plt.ylabel('Score Values')
plt.title("AMI and NMI results for different rho values(estimated beta_3)")
#plt.annotate(f'Variance: {rho_variance:.3f}', xy=(1.02, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
#plt.annotate(f'Standard Deviation: {rho_stand_deviation:.3f}', xy=(1.05, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
plt.annotate(f'Max_AMI: {max_ami_results}', xy=(rho_value[max_index], max_ami_results), xytext=(-10, -20), textcoords='offset points')
plt.annotate(f'Max_NMI: {max_nmi_results}', xy=(rho_value[max_nmi_index], max_nmi_results), xytext=(-10, -70), textcoords='offset points')
plt.legend()
plt.show()
