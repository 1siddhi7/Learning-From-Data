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

m = np.load('india_map.npy')
H, W = m.shape
data_original = np.array(list(np.ndindex((H,W))))
labels_original = m.flatten()
labels_original = 1*labels_original
labels_original[labels_original == 0] = -1

#Subsample data
N = 15000
idx = np.random.choice(len(data_original), size=N)
data = data_original[idx]
labels = labels_original[idx]
x_train, x_test, y_train, y_test = train_test_split(data, labels,test_size=0.20)

#Scaling th features
scaler = StandardScaler() 
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)
scaled_data_original = scaler.transform(data_original)

#Using 5-fold cross validation to find the best value of gamma for SVM using rbf kernel
gammas = np.linspace(-7,5,14)
cv_scores = []
test_scores = []
train_scores = []

for item in 10**gammas:
    print("gamma =", item)
    
    svclassifier = SVC(C = 5, kernel='rbf', gamma = item) 
     
    scores = cross_val_score(svclassifier, scaled_x_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

    svclassifier.fit(scaled_x_train, y_train)
    
    y_pred = svclassifier.predict(scaled_x_test) 
    y_pred_train = svclassifier.predict(scaled_x_train)    
    test_scores.append(metrics.accuracy_score(y_test, y_pred))
    train_scores.append(metrics.accuracy_score(y_train, y_pred_train))

#plotting decision boundary for best value of gamma
i = np.argmax(cv_scores)
gamma = 10**gammas[i]
print("best gamma =", gamma)
print("plotting decision boundary")

svclassifier_best = SVC(C = 5, kernel='rbf', gamma = gamma) 
svclassifier_best.fit(scaled_x_train, y_train)
sp_scaled = svclassifier_best.support_vectors_
sp = scaler.inverse_transform(sp_scaled)
data_pred = svclassifier_best.predict(scaled_data_original) 
print("Accuracy on entire data:",metrics.accuracy_score(labels_original, data_pred ))
idx_green = np.where(data_pred == 1)
idx_red = np.where(data_pred == -1)

data_green = data_original[idx_green]
data_red = data_original[idx_red]

    
plt.scatter(data_green[:, 0], data_green[:, 1], color='green', s=2,label="output : in India")
plt.scatter(data_red[:, 0], data_red[:, 1], color='red', s=2,label="output : out of India")
plt.scatter(sp[:, 0], sp[:, 1], color='black', s=2,label="support vectors")

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
