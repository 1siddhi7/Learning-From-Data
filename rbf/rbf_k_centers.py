import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster.k_means_ import _k_init
from sklearn.metrics import pairwise_distances_argmin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn import metrics
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state
import scipy.sparse as sp


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

#Scaling of features
scaler = StandardScaler() 
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)
scaled_data_original = scaler.transform(data_original)


random_state = 0
K = 5896
gamma = 177.82
random_state = check_random_state(random_state)
x_squared_norms = row_norms(scaled_x_train, squared=True)

if not sp.issparse(scaled_x_train):
        scaled_x_train_mean = scaled_x_train.mean(axis=0)
        scaled_x_train -= scaled_x_train_mean
        
if not sp.issparse(scaled_x_test):
        scaled_x_test_mean = scaled_x_test.mean(axis=0)
        scaled_x_test -= scaled_x_test_mean
        
if not sp.issparse(scaled_data_original):
        scaled_data_original_mean = scaled_data_original.mean(axis=0)
        scaled_data_original -= scaled_data_original_mean
        
#Initializing the centers using k-means++ algorithm implementation of sklearn
centers = _k_init(scaled_x_train, K, random_state=random_state, x_squared_norms=x_squared_norms)


def find_centers(X, n_clusters, centers ):
    
    ''' Function to find centers using lloyd's algorithm.
        paramaeters to be passed: 1. data for which centers are to be found,
                                  2. number of centers & 3. initial centers'''
    
    centers = centers
    K = np.arange(n_clusters)
    i = 0
    
    while True:
        print("Iteration: ", i)
        i = i + 1
        
        # Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        empty = np.setdiff1d(K,labels).astype('int')
                
        #  Find new centers from means of points
        append = centers[empty]
        new_centers = np.array([X[labels == i].mean(0)
                                for i in np.unique(labels)])
        new_centers = np.concatenate((new_centers,append))
        # Check for convergence
        if np.all(centers == new_centers):
            print('breaking')
            break
        centers = new_centers
        
    return centers, labels

scaled_centers, labels = find_centers(scaled_x_train, K, centers)

phi = scaled_x_train[:,np.newaxis,:] - scaled_centers[np.newaxis,:,:]
phi = np.sum(phi**2, axis=2)
phi = np.exp( -gamma * phi)

#Calculating weights vector
x_d = np.matmul(np.linalg.inv(np.matmul(phi.T,phi)),phi.T)
w = np.matmul(x_d,y_train)

#Plotting decision boundary
def hypothesis(x):
    ans = 0
    for i in range(K):
        ans += w[i]*np.exp(-gamma * np.linalg.norm(x - scaled_centers[i])**2)
    return ans

pred = np.zeros(data_original.shape[0])

for i in range(data_original.shape[0]):
    print(i)
    pred[i] = np.sign(hypothesis(scaled_data_original[i]))
print("Accuracy on entire data:",metrics.accuracy_score(labels_original, pred ))

idx_green = np.where(pred == 1)
idx_red = np.where(pred == -1)

data_green = data_original[idx_green]
data_red = data_original[idx_red]

    
plt.scatter(data_green[:, 1], data_green[:, 0], color='green', s=2,label="output : in India")
plt.scatter(data_red[:, 1], data_red[:, 0], color='red', s=2,label="output : out of India")
plt.scatter(sp[:, 1], sp[:, 0], color='black', s=2,label="support vectors")
plt.gca().invert_yaxis()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
