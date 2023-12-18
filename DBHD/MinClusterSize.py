from LDClusAlgo import LDClus
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
#print(len(X))

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

#print(len(X))
print(len(Y))


rho, beta = (1.8, 0.0)

'''
minClusterSize_1 = math.log2(len(X)) + 5
#print(minClusterSize_1) 
estimate_n_1 = round(minClusterSize_1)
#print(estimate_n_1)
y_pred = LDClus(X, estimate_n_1, rho, beta)


minClusterSize_2 = math.log2(len(X)^2)
estimate_n_2 = round(minClusterSize_2)
y_pred = LDClus(X, estimate_n_2, rho, beta)
#print(minClusterSize_2)

minClusterSize_3 = math.log(len(X)^2)/math.log(4)
estimate_n_3 = round(minClusterSize_3)
y_pred = LDClus(X, estimate_n_3, rho, beta)
#print(minClusterSize_3)
'''

results = []
cluster_size = []
#index = 0
for n in np.arange(5, 200, 1):
    cluster_size.append(n)
    y_pred = LDClus(X, n, rho, beta) #return Labels
    print(len(y_pred))
    #print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
    results.append(adjusted_mutual_info_score(Y, y_pred))


n_variance = np.var(results)
n_stand_deviation = np.std(results)
#print(beta_variance)
#print(beta_stand_deviation)
#print(results)


'''
plt.scatter(cluster_size, results)
plt.xlabel('minimum cluster size')
plt.ylabel('AMI results')
plt.title("AMI results for different cluster size values")
plt.annotate(f'Variance: {n_variance:.3f}', xy=(1.02, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
plt.annotate(f'Standard Deviation: {n_stand_deviation:.3f}', xy=(1.05, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
plt.legend()
plt.show()
'''


'''
X_column1 = [x[0] for x in X]
X_column2 = [x[1] for x in X]

plt.scatter(X_column1, X_column2, c = Y)
plt.xlabel('X Column 1')
plt.ylabel('X Column 2')
#plt.colorbar(label='Labels')
plt.show()
'''
