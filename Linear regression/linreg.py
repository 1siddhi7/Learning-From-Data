import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
boston = load_boston()

outputs = boston.target
data = boston.data

data=np.insert(data, 0, 1, axis=1)

x_train, x_test, y_train, y_test = train_test_split(data, outputs, test_size=0.20)

x_d = np.matmul(np.linalg.inv(np.matmul(np.transpose(x_train),x_train)),np.transpose(x_train))
w = np.matmul(x_d,y_train)

yp=np.matmul(x_test,w)
plot_x=list(y_test)
plot_y=list(yp)

plt.scatter(plot_x, plot_y,color= "green",  marker= "*",s=100)
plt.plot([0,50],[0,50], color='black', linewidth=1,label='reference line y=x')
plt.ylabel('predicted price',fontsize=15)
plt.xlabel('actual price',fontsize=15)
plt.title('predicted price vs actual price',fontsize=15,fontweight='bold')
plt.legend(fontsize=12)
plt.show()

def plot_feature(k):

    x=[]
    y=[]
    yp=[]
    for i in range(len(x_test)):
        x.append(x_test[i][k])
        y.append(y_test[i])
    N=len(x_test)
    sum=0
    for i in range(N):
        sum=sum+(y_test[i] - w[k]*x_test[i][k])

    b = (1/N)*sum
    for i in range(N):
        yp.append(w[k]*x_test[i][k] + b)

    plt.plot(x, y, color='green', linewidth = 0, marker='o', markerfacecolor='green', markersize=3,label="actual")
    plt.plot(x, yp, color='black', linewidth = 2, marker='o',markerfacecolor='black', markersize=0,label="predicted")
    plt.xlabel(boston.feature_names[k-1],fontsize=15)
    plt.ylabel("MEDV",fontsize=15)
    plt.legend(fontsize=12)
    plt.show()


plot_feature(5)
plot_feature(6)
plot_feature(8)
plot_feature(11)