#from LDClusAlgo import LDClus
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt
import arff
import pandas as pd

with open('/Users/xueni/Desktop/DBHD/data/cluto-t4-8k.arff', 'r') as f:
    arff_data = arff.load(f)

data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])

data['CLASS'] = data['CLASS'].apply(lambda x: -1 if x == 'noise' else int(x) if x != '' else None)
#data['CLASS'] = data['CLASS'].astype(int) 

X = data['x']
y = data['y']
class_label = data['CLASS']


plt.figure(figsize=(8, 6))
plt.scatter(X, y, c = class_label)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cluto-t4-8k')
#plt.colorbar(label='Class')
plt.grid(True)
plt.show()


