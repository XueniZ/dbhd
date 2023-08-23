from LDClusAlgo import LDClus
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
import arff
import umap
import math
import sys
from sklearn.preprocessing import LabelEncoder

sys.setrecursionlimit(1000000)

with open('/Users/xueni/Desktop/DBHD/data/heart-statlog.arff', 'r') as f:
    arff_data = arff.load(f)

data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])
data_without_missing_value = data.dropna(subset=[data.columns[-2]])

features = data_without_missing_value.iloc[:, :-1]  # Features (attributes)
class_label = data_without_missing_value.iloc[:, -1]   # Class labels

features_array = features.values

#data['class'] = data['class'].astype(int)
label_encoder = LabelEncoder()
labels_numeric = label_encoder.fit_transform(class_label)

#Using Dimensionality Reduction
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(features)


n, rho = 95, 1.8

#rho = 1.8

#erste Formel
minClusterSize_1 = math.log2(len(features_array)) + 5
#print(minClusterSize_1) 
estimate_n_1 = round(minClusterSize_1)


#zweite Formel
minClusterSize_2 = math.log2(len(features_array)^2)
estimate_n_2 = round(minClusterSize_2)

#dritte Formel
minClusterSize_3 = math.log(len(features_array)^2)/math.log(4)
estimate_n_3 = round(minClusterSize_3)


#vierte Formel
minClusterSize_4 = 2 * features.shape[1]

results_ami = []
results_nmi = []
beta_value = []
#index = 0
for beta in np.arange(0, 1, 0.01):
    #beta = beta / 100.0
    beta_value.append(beta)
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


plt.plot(beta_value, results_ami, c = 'blue', label = 'AMI Score')
plt.plot(beta_value, results_nmi, c = 'red', label = 'NMI Score')
plt.title('AMI and NMI results with different beta values')
plt.xlabel('beta value(0-1)')
plt.ylabel('Score Values')
plt.annotate(f'Max_AMI: {max_ami_results}', xy=(beta_value[max_ami_index], max_ami_results), xytext=(10, -20), textcoords='offset points')
plt.annotate(f'Max_NMI: {max_nmi_results}', xy=(beta_value[max_nmi_index], max_nmi_results), xytext=(-50, -50), textcoords='offset points')
plt.legend()
plt.show()

'''
beta_variance = np.var(results)
beta_stand_deviation = np.std(results)

plt.scatter(beta_value, results)
plt.xlabel('beta')
plt.ylabel('AMI results')
plt.title("AMI results for n = log(X.size^2) in different beta values(distance: 0.01)")
plt.annotate(f'Variance: {beta_variance:.3f}', xy=(1.02, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
plt.annotate(f'Standard Deviation: {beta_stand_deviation:.3f}', xy=(1.05, 0.5), xycoords='axes fraction', rotation=90, ha='left', va='center')
plt.show()
'''