from LDClusAlgo import LDClus
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
import arff
import umap
import math
from sklearn.preprocessing import LabelEncoder
from beta import estimated_beta
from beta import find_sigma
from beta import find_epsilon

with open('/Users/xueni/Desktop/DBHD/data/heart-statlog.arff', 'r') as f:
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

rho, beta = (1.8, 0.0)
results_ami = []
results_nmi = []
cluster_size = []
#index = 0
for n in np.arange(5, 200, 1):
    cluster_size.append(n)
    
    sigma = find_sigma(features_array, n)
    eps = find_epsilon(features_array, n)
    esti_beta = estimated_beta(features_array, sigma, eps, n)

    y_pred = LDClus(features_array, n, rho, beta) #return Labels
    #print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
    results_ami.append(adjusted_mutual_info_score(class_label, y_pred))
    results_nmi.append(normalized_mutual_info_score(class_label, y_pred))

n_variance = np.var(results_ami)
n_stand_deviation = np.std(results_ami)
n_variance = np.var(results_nmi)
n_stand_deviation = np.std(results_nmi)

max_ami_results = max(results_ami)
max_ami_index = results_ami.index(max_ami_results)

max_nmi_results = max(results_nmi)
max_nmi_index = results_nmi.index(max_nmi_results)


plt.plot(cluster_size, results_ami, c = 'blue', label = 'AMI Score')
plt.plot(cluster_size, results_nmi, c = 'red', label = 'NMI Score')
plt.title('AMI and NMI results with different cluster sizes')
plt.xlabel('cluster size(5 to 200)')
plt.ylabel('Score Values')
plt.annotate(f'Max_AMI: {max_ami_results}', xy=(cluster_size[max_ami_index], max_ami_results), xytext=(10, -20), textcoords='offset points')
plt.annotate(f'Max_NMI: {max_nmi_results}', xy=(cluster_size[max_nmi_index], max_nmi_results), xytext=(70, -70), textcoords='offset points')
plt.legend()
plt.show()

'''
plt.scatter(cluster_size, results_ami)
plt.xlabel('minimum cluster size')
plt.ylabel('AMI results')
plt.title("AMI results for different cluster size values")
plt.annotate(f'Variance: {n_variance:.3f}', xy=(1.02, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
plt.annotate(f'Standard Deviation: {n_stand_deviation:.3f}', xy=(1.05, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
plt.legend()
plt.show()
'''