#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:46:10 2019

@author: siddhi
"""

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn import metrics



m = np.load('india_map.npy')
H, W = m.shape
data_original = np.array(list(np.ndindex((H,W))))
labels_original = m.flatten()
labels_original = 1*labels_original
labels_original[labels_original == 0] = -1

#Subsample data
N = 25000   
idx = np.random.choice(len(data_original), size=N)
data = data_original[idx]
labels = labels_original[idx]
x_train, x_test, y_train, y_test = train_test_split(data, labels,test_size=0.20)

scaler = StandardScaler() 
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)
scaled_data_original = scaler.transform(data_original)

K = 5896
kmeans = KMeans(n_clusters = K).fit(scaled_x_train)


scaled_centers = kmeans.cluster_centers_
centers = scaler.inverse_transform(scaled_centers)

phi = np.zeros((x_train.shape[0], K))
gamma = 177.82

for i in range(x_train.shape[0]):
    for j in range(K):
        phi[i][j] = np.exp(-gamma*(np.linalg.norm(scaled_x_train[i] - scaled_centers[j]))**2)
        
x_d = np.matmul(np.linalg.inv(np.matmul(phi.T,phi)),phi.T)
w = np.matmul(x_d,y_train)

def hypothesis(x):
    ans = 0
    for i in range(K):
        ans += w[i]*np.exp(-gamma * np.linalg.norm(x - scaled_centers[i])**2)
    return ans

pred = np.zeros(data_original.shape[0])

for i in range(data_original.shape[0]):
    pred[i] = np.sign(hypothesis(scaled_data_original[i]))
print("Accuracy on entire data:",metrics.accuracy_score(labels_original, pred ))

idx_green = np.where(pred == 1)
idx_red = np.where(pred == -1)

data_green = data_original[idx_green]
data_red = data_original[idx_red]

    
plt.scatter(data_green[:, 0], data_green[:, 1], color='green', s=2,label="output : in India")
plt.scatter(data_red[:, 0], data_red[:, 1], color='red', s=2,label="output : out of India")

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


