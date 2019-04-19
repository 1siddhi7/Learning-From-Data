#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:23:40 2019

@author: siddhi
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

features=[]
labels=[]

with open("marks.txt",'r') as f:
    
    for line in f:   
        
        if not line.strip():
            continue
        marks1,marks2,result=line.strip().split(',')
        
        labels.append(int(result))
        features.append([1,float(marks1),float(marks2)])
        
features=np.array(features)
labels=np.array(labels)


x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
#print(x_train.shape,y_train.shape)
#N = x_train.shape[0]
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

#theta=np.zeros(3)
#learn_rate = 0.1

def logistic_func(w, x): 
    ''' 
    logistic(sigmoid) function 
    '''
    return 1.0/(1 + np.exp(-np.dot(x, w.T))) 

#w=np.array([[0,0,0]])
#print(logistic_func(w,x_train))
def log_gradient(w, x, y): 
    ''' 
    logistic gradient function 
    '''
    first_calc = logistic_func(w, x) - y.reshape(x.shape[0], -1) 
    final_calc = np.dot(first_calc.T, x) 
    return final_calc 
  
#print(log_gradient(np.array([[0,0,0]]),x_train,y_train))

def cost_func(w, x, y): 
    ''' 
    cost function, J 
    '''
    log_func_vec = logistic_func(w, x) 
    y = np.squeeze(y) 
    step1 = y * np.log(log_func_vec) 
    step2 = (1 - y) * np.log(1 - log_func_vec) 
    final = -step1 - step2 
    return np.mean(final) 
  
#print(cost_func(w,x_train,y_train))
def grad_desc(x, y, w, lr=0.01, converge_change=.001): 
    ''' 
    gradient descent function 
    '''
    cost = cost_func(w, x, y) 
    change_cost = 1
    num_iter = 1
    c = [cost] 
    n = [num_iter]
    while(change_cost > converge_change): 
        old_cost = cost 
        w = w - (lr * log_gradient(w, x, y)) 
        cost = cost_func(w, x, y) 
        change_cost = old_cost - cost 
        num_iter = num_iter + 1
      
        c.append(cost)
        n.append(num_iter)
        
    plt.plot(n, c, color='green', linewidth = 2, marker='o', markerfacecolor='blue', markersize=4) 

    plt.ylabel('error')
    plt.xlabel('no of iterations')
    plt.title('cost vs no. of iterations')
    plt.show()    

    return w, num_iter  
  
  
def pred_values(w, x): 
    ''' 
    function to predict labels 
    '''
    pred_prob = logistic_func(w, x) 
    pred_value = np.where(pred_prob >= 0.5, 1, 0) 
    return np.squeeze(pred_value) 
  
  

      
 
weights = np.matrix(np.zeros(x_train.shape[1])) 

weights, num_iter = grad_desc(x_train, y_train, weights)

print("Estimated regression coefficients:", weights) 
print("No. of iterations:", num_iter) 

y_pred = pred_values(weights, x_test)
#print(y_pred)
#print(y_test)

print("Correctly predicted labels:", np.sum(y_test == y_pred)) 
  
