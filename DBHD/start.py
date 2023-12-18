from LDClusAlgo import LDClus
from sklearn.datasets import load_digits
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt

X, Y = load_digits(return_X_y=True) #dataSet, labels
#print(X)

#print(Y)

idx = np.random.permutation(len(X))
#print(idx)
X, Y = X[idx], Y[idx]

print(X)
print(Y)

n, rho, beta = (95, 1.8, 0.0) # minclusterSize, rho, beta paramter

y_pred = LDClus(X, n, rho, beta) #return Labels
print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')

