import numpy as np
from sklearn.model_selection import train_test_split
from scipy import spatial
from scipy import stats
import matplotlib.pyplot as plt


def calculate_bias_variance(x_train, y_train, x_test, y_test, k):
    '''
    Creates multiple splits of the data and returns K-NN bias and variance
    '''
    N = 10
    predict = np.zeros((N, len(y_test)))
    
    for i in range(N):
        xi_train, xi_test, yi_train, yi_test = train_test_split(x_train,
                                                                y_train,
                                                                test_size=0.20)
        print('Iteration: {:d}'.format(i))
        
        tree = spatial.KDTree(xi_train)
        nearest_dist, nearest_ind = tree.query(x_test, k=k, p=2)
        options = yi_train[nearest_ind]
        ans = stats.mode(options, axis=1)[0].flatten()
        predict[i] = ans
        
    bias = y_test - np.mean(predict, axis=0)
    variance = np.var(predict, axis=0)
    return np.mean(bias), np.mean(variance)


def plot_bias_variance(x_train, y_train, x_test, y_test):
    N_iterations = 10
    k_plot = []
    b_plot = []
    v_plot = []
    
    for i in range(N_iterations):
        k = (i+1)*20
        print('Calculating bias and variance for k = {:d}'.format(k))
        k_plot.append(k)
        b, v = calculate_bias_variance(x_train, y_train, x_test, y_test, k=k)
        b_plot.append(b)
        v_plot.append(v)
        
    plt.plot(k_plot, b_plot, color='green', linewidth=2, marker='o',
             markerfacecolor='green', markersize=8, label = "bias") 
    plt.plot(k_plot, v_plot, color='blue', linewidth=2, marker='o',
             markerfacecolor='blue', markersize=8, label = "variance") 
    plt.xlabel('k') 
    plt.title('Bias and variance vs. k') 
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    m = np.load('india_map.npy')
    H, W = m.shape
    
    data = np.array(list(np.ndindex((H,W))))
    labels = m.flatten()
    
    # subsample data
    N = 10000
    idx = np.random.choice(len(data), size=N)
    data = data[idx]
    labels = labels[idx]
    
    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.20)
    print('Size of train data = {:d}, size of test data = {:d}'.
          format(len(y_train), len(y_test)))
    
    plot_bias_variance(x_train, y_train, x_test, y_test)
