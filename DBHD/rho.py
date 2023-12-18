from LDClusAlgo import LDClus
from sklearn.datasets import load_digits
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt
import sys
import math

sys.setrecursionlimit(1000000)

dataset_path = "/Users/xueni/Desktop/DBHD/data/    artificial_localN/artificial_localN_data.txt"
labels_path = "/Users/xueni/Desktop/DBHD/data/    artificial_localN/artificial_localN_label.txt"

with open(dataset_path, 'r') as file:
    dataset_lines = file.readlines()

X = []

for line in dataset_lines:
    values = line.strip().split()
    row = [float(value) for value in values]
    X.append(row)

X = np.array(X)
#print(X)


with open(labels_path, 'r') as file:
    labels_lines = file.readlines()

Y = []

for line in labels_lines:
    label = int(line.strip())
    Y.append(label)

Y = np.array(Y)
#print(Y)


idx = np.random.permutation(len(X))

X = X[idx]
Y = Y[idx]


'''
X, Y = load_digits(return_X_y=True) #dataSet, labels
idx = np.random.permutation(len(X))
X, Y = X[idx], Y[idx]
n, rho, beta = (95, 1.8, 0.0) # minclusterSize, rho, beta paramter
y_pred = LDClus(X, n, rho, beta) #return Labels
print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
'''

minClusterSize_1 = math.log2(len(X)) + 5
#print(minClusterSize_1) 
estimate_n_1 = round(minClusterSize_1)
#print(estimate_n_1)

beta = 0.0
results = []
rho_value = []
#index = 0
for rho in np.arange(0.1, 2, 0.01):
    rho_value.append(rho)
    y_pred = LDClus(X, estimate_n_1, rho, beta) #return Labels
    #print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
    results.append(adjusted_mutual_info_score(Y, y_pred))

rho_variance = np.var(results)
rho_stand_deviation = np.std(results)
#print(beta_variance)
#print(beta_stand_deviation)
#print(results)

plt.plot(rho_value, results, color = 'green')
plt.xlabel('rho')
plt.ylabel('AMI results')
plt.title("AMI results for different rho values(distance: 0.01)")
plt.annotate(f'Variance: {rho_variance:.3f}', xy=(1.02, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
plt.annotate(f'Standard Deviation: {rho_stand_deviation:.3f}', xy=(1.05, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
plt.show()


