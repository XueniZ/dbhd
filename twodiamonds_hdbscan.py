import hdbscan
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arff
from sklearn.neighbors import NearestNeighbors

with open('/Users/xueni/Desktop/DBHD/data/twodiamonds.arff', 'r') as f:
    arff_data = arff.load(f)


data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])

data['class'] = data['class'].astype(int)

X = data.iloc[:, :-1]  # Features (attributes)
y = data.iloc[:, -1]   # Class labels
X_array = X.values

results_ami = []
results_nmi = []
cluster_size = []



for n in np.arange(5, 200, 1):
    n = int(n)
    cluster_size.append(n)
    hdb = hdbscan.HDBSCAN(min_cluster_size=n, min_samples=5).fit(X_array)
    y_pred = hdb.labels_
    #print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
    results_ami.append(adjusted_mutual_info_score(y, y_pred))
    results_nmi.append(normalized_mutual_info_score(y, y_pred))

max_ami_results = max(results_ami)
max_index = results_ami.index(max_ami_results)

max_nmi_results = max(results_nmi)
max_nmi_index = results_nmi.index(max_nmi_results)

plt.plot(cluster_size, results_ami, c = 'blue', label = 'AMI Score')
plt.plot(cluster_size, results_nmi, c = 'red', label = 'NMI Score')
plt.xlabel('cluster size(5 to 200)')
plt.ylabel('Score Values')
plt.title("AMI and NMI results for different cluster size")
plt.annotate(f'Max_AMI: {max_ami_results}', xy=(cluster_size[max_index], max_ami_results), xytext=(10, -20), textcoords='offset points')
plt.annotate(f'Max_NMI: {max_nmi_results}', xy=(cluster_size[max_nmi_index], max_nmi_results), xytext=(10, -50), textcoords='offset points')
plt.legend()
plt.show()
