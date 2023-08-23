from LDClusAlgo import LDClus
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt
import arff

with open('/Users/xueni/Desktop/DBHD/data/twenty.arff', 'r') as f:
    arff_data = arff.load(f)


data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])

data['class'] = data['class'].astype(int)

X = data.iloc[:, :-1]  # Features (attributes)
y = data.iloc[:, -1]   # Class labels
X_array = X.values

'''
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c = class_label, cmap = 'tab20')
plt.xlabel('x')
plt.ylabel('y')
plt.title('twenty')
plt.colorbar(label='Class')
plt.grid(True)
plt.show()
'''

results = []
beta_value = []
rho_value = []
cluster_size = []

for beta in np.arange(0, 1, 0.01):
    for n in np.arange(5, 200, 1):
        for rho in np.arange(0.1, 2, 0.1):
            rho_value.append(rho)
            cluster_size.append(n)
            beta_value.append(beta)
            y_pred = LDClus(X_array, n, rho, beta) #return Labels
            #print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
            results.append(adjusted_mutual_info_score(y, y_pred))

data = np.array([beta_value], [cluster_size], [rho_value])

plt.plot(data.T, results)
plt.legend()
plt.show()
