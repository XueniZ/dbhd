import hdbscan
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arff
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import umap

with open('/Users/xueni/Desktop/DBHD/data/dermatology.arff', 'r') as f:
    arff_data = arff.load(f)

data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])
data_without_missing_value = data.dropna(subset=[data.columns[-2]])

features = data_without_missing_value.iloc[:, :-1]  # Features (attributes)
class_label = data_without_missing_value.iloc[:, -1]   # Class labels

features_array = features.values

label_encoder = LabelEncoder()
labels_numeric = label_encoder.fit_transform(class_label)

reducer = umap.UMAP(n_components=2, random_state = 42)
embedding = reducer.fit_transform(features)

results_ami = []
results_nmi = []
cluster_size = []

for n in np.arange(5, 200, 1):
    n = int(n)
    cluster_size.append(n)
    hdb = hdbscan.HDBSCAN(min_cluster_size=n, min_samples=4).fit(features_array)
    y_pred = hdb.labels_
    #print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
    results_ami.append(adjusted_mutual_info_score(class_label, y_pred))
    results_nmi.append(normalized_mutual_info_score(class_label, y_pred))

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
