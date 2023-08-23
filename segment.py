from LDClusAlgo import LDClus
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arff
import umap
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D 

with open('/Users/xueni/Desktop/DBHD/data/segment.arff', 'r') as f:
    arff_data = arff.load(f)

data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])

features = data.iloc[:, :-1]  # Features (attributes)
class_label = data.iloc[:, -1]   # Class labels

#data['class'] = data['class'].astype(int)
label_encoder = LabelEncoder()
labels_numeric = label_encoder.fit_transform(class_label)

#Using Dimensionality Reduction
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(features)


#create a 2D plot
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c = labels_numeric, s = 5)
plt.colorbar(label = 'class_label')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('UMAP Projection of Image Segment Data')
plt.grid(True)
plt.show()


'''
#create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels_numeric, s=5)
ax.set_title('UMAP Projection of Image Segment Data(3D)')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.show()

'''