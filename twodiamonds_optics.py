from sklearn.cluster import OPTICS
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arff
from sklearn.neighbors import NearestNeighbors
import math

with open('/Users/xueni/Desktop/DBHD/data/twodiamonds.arff', 'r') as f:
    arff_data = arff.load(f)


data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])

data['class'] = data['class'].astype(int)

X = data.iloc[:, :-1]  # Features (attributes)
y = data.iloc[:, -1]   # Class labels
X_array = X.values

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

results_ami = []
results_nmi = []
eps_value = []
samples = []

'''
for sample in np.arange(1, 800, 1):
    samples.append(sample)
    optics = OPTICS(min_samples = sample).fit(X_array)
    y_pred = optics.labels_
    results_ami.append(adjusted_mutual_info_score(y, y_pred))
    results_nmi.append(normalized_mutual_info_score(y, y_pred))
'''

for eps in np.arange(0, 1, 0.01):
    eps_value.append(eps)
    optics = OPTICS(min_samples = estimate_n_1, eps = eps).fit(X_array)
    y_pred = optics.labels_
    results_ami.append(adjusted_mutual_info_score(y, y_pred))
    results_nmi.append(normalized_mutual_info_score(y, y_pred))


max_ami_results = max(results_ami)
max_index = results_ami.index(max_ami_results)

max_nmi_results = max(results_nmi)
max_nmi_index = results_nmi.index(max_nmi_results)
'''
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

plt.plot(eps_value, results_ami, c = 'blue', label = 'AMI Score')
plt.plot(eps_value, results_nmi, c = 'red', label = 'NMI Score')
plt.xlabel('epsilon value')
plt.ylabel('Score Values')
plt.title("AMI and NMI results for different epsilon values")
plt.annotate(f'Max_AMI: {max_ami_results}', xy=(eps_value[max_index], max_ami_results), xytext=(10, -20), textcoords='offset points')
plt.annotate(f'Max_NMI: {max_nmi_results}', xy=(eps_value[max_nmi_index], max_nmi_results), xytext=(10, -50), textcoords='offset points')
plt.legend()
plt.show()