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
from read_file import read_file

'''
with open('/Users/xueni/Desktop/DBHD/data/dermatology.arff', 'r') as f:
    arff_data = arff.load(f)

data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])
data_without_missing_value = data.dropna(subset=[data.columns[-2]])

features = data_without_missing_value.iloc[:, :-1]  # Features (attributes)
class_label = data_without_missing_value.iloc[:, -1]   # Class labels

features_array = features.values
'''

X, y = read_file('dermatology', 1)
from sklearn.preprocessing import MinMaxScaler
m = MinMaxScaler()
X = m.fit_transform(X)

label_encoder = LabelEncoder()
labels_numeric = label_encoder.fit_transform(y)

reducer = umap.UMAP(n_components=2, random_state = 42)
embedding = reducer.fit_transform(X)

cluster_size = np.arange(5, 200, 5)
rho_value = np.arange(0.05, 2.05, 0.05)
beta_value = np.arange(0, 0.7, 0.1)

#erste Formel
minClusterSize_1 = math.log2(len(X)) + 5
#print(minClusterSize_1) 
estimate_n_1 = round(minClusterSize_1)


#zweite Formel
minClusterSize_2 = math.log2(len(X)^2)
estimate_n_2 = round(minClusterSize_2)


#dritte Formel
minClusterSize_3 = math.log2(len(X))
estimate_n_3 = round(minClusterSize_3)


#vierte Formel
estimate_n_4= 2 * X.shape[1]

#fünfte Formel
minClusterSize_5 = math.log2(len(X)) + X.shape[1]
estimate_n_5 = round(minClusterSize_5)


'''
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

    y_pred = LDClus(features_array, n, rho, esti_beta) #return Labels
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
plt.title('AMI and NMI results with different cluster sizes(estimated beta_3)')
plt.xlabel('cluster size(5 to 200)')
plt.ylabel('Score Values')
plt.annotate(f'Max_AMI: {max_ami_results}', xy=(cluster_size[max_ami_index], max_ami_results), xytext=(10, -20), textcoords='offset points')
plt.annotate(f'Max_NMI: {max_nmi_results}', xy=(cluster_size[max_nmi_index], max_nmi_results), xytext=(70, -70), textcoords='offset points')
plt.legend()
plt.show()
'''
'''
fig, axs = plt.subplots(5, 4, figsize=(12, 15))

for i in range(5):
    for j in range(4):
        rho = rho_value[i * 4 + j]
        ami_scores = []
        nmi_scores = []
        for beta in beta_value:
            y_pred = LDClus(features, estimate_n_2, rho, beta) #return Labels
        
            ami = adjusted_mutual_info_score(class_label, y_pred)
            nmi = normalized_mutual_info_score(class_label, y_pred)
        
            ami_scores.append(ami)
            nmi_scores.append(nmi)
    
        axs[i, j].plot(beta_value, ami_scores, label='AMI')
        axs[i, j].plot(beta_value, nmi_scores, label='NMI')
        axs[i, j].set_title(f'Rho = {rho: .1f}')
        axs[i, j].set_ylabel('Scores')
        axs[i, j].legend()

for ax in axs.flat:
    ax.label_outer()

plt.xlabel('Beta Values')
plt.tight_layout()
plt.show()
'''


results_ami = []
results_nmi = []
'''
for n in cluster_size:
    for rho in rho_value:
        for beta in beta_value:
            y_pred = LDClus(X, n, rho, beta) #return Labels
            #print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
            results_ami.append(adjusted_mutual_info_score(y, y_pred))
            results_nmi.append(normalized_mutual_info_score(y, y_pred))
    

max_ami_results = max(results_ami)
max_ami_index = results_ami.index(max_ami_results)

max_nmi_results = max(results_nmi)
max_nmi_index = results_nmi.index(max_nmi_results)

print(max_ami_results)
print(max_nmi_results)
'''

for rho in rho_value:
    for beta in beta_value:
        y_pred = LDClus(X, estimate_n_5, rho, beta) #return Labels
        #print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')
        results_ami.append(adjusted_mutual_info_score(y, y_pred))
        results_nmi.append(normalized_mutual_info_score(y, y_pred))
    

max_ami_results = max(results_ami)
max_ami_index = results_ami.index(max_ami_results)

max_nmi_results = max(results_nmi)
max_nmi_index = results_nmi.index(max_nmi_results)

print(max_ami_results)
print(max_nmi_results)