#from LDClusAlgo import LDClus
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt
import arff
import pandas as pd

with open('/Users/xueni/Desktop/DBHD/data/twodiamonds.arff', 'r') as f:
    arff_data = arff.load(f)

data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])

data['class'] = data['class'].astype(int)

x = data['x']
y = data['y']
class_label = data['class']


plt.figure(figsize=(8, 6))
plt.scatter(x, y, c = class_label)
plt.xlabel('x')
plt.ylabel('y')
plt.title('two diamonds')
#plt.colorbar(label='Class')
plt.grid(True)
plt.show()


