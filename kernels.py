#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 19:51:25 2019

@author: siddhi
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics.pairwise import euclidean_distances

#from qpsolvers import solve_qp

m = np.load('india_map.npy')
H, W = m.shape
data_original = np.array(list(np.ndindex((H,W))))
labels_original = m.flatten()
labels_original = 1*labels_original
labels_original[labels_original == 0] = -1

print(data_original.shape, labels_original.shape)
print(data_original)
print(labels_original)

N = 20000
idx = np.random.choice(len(data_original), size=N)
data = data_original[idx]
labels = labels_original[idx]
x_train, x_test, y_train, y_test = train_test_split(data, labels,test_size=0.20)

scaler = StandardScaler() 
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)
scaled_data_original = scaler.transform(data_original)

gammas = 10**np.linspace(1,4,6)

def gaussian_kernel(X, Y):
	    kernel = euclidean_distances(X, Y) ** 2
	    kernel = kernel*(-150)
	    kernel = np.exp(kernel)
	    return kernel



cv_scores = []


for item in gammas:
    print("gamma =", item)
    
    svclassifier = SVC(C = 5, kernel='rbf', gamma = item) 
     
    scores = cross_val_score(svclassifier, scaled_x_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

    svclassifier.fit(scaled_x_train, y_train)
    
    y_pred = svclassifier.predict(scaled_x_test) 
    print(y_pred.shape) 
    print("Accuracy on training data:",metrics.accuracy_score(y_test, y_pred))


i = np.argmax(cv_scores)
gamma = gammas[i]
print("best gamma =", gamma)
print("plotting decision boundary")

svclassifier_best = SVC(C = 5, kernel='rbf', gamma = gamma) 
svclassifier_best.fit(scaled_x_train, y_train)
data_pred = svclassifier_best.predict(scaled_data_original) 
print("Accuracy:",metrics.accuracy_score(labels_original, data_pred ))
idx_green = np.where(data_pred == 1)
idx_red = np.where(data_pred == -1)

data_green = data_original[idx_green]
data_red = data_original[idx_red]

    
plt.plot(data_green[:, 0], data_green[:, 1], color='green', linewidth = 0, marker='o', 
         markerfacecolor='green', markersize=2,label="output : in India")
plt.plot(data_red[:, 0], data_red[:, 1], color='red', linewidth = 0, marker='o', 
         markerfacecolor='red', markersize=2,label="output : out of India")

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
