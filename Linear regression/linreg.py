import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
boston = load_boston()

outputs=boston.target
data=boston.data

data=np.insert(data, 0, 1, axis=1)

x_train, x_test, y_train, y_test = train_test_split(data, outputs, test_size=0.20)

x_d= np.matmul(np.linalg.inv(np.matmul(np.transpose(x_train),x_train)),np.transpose(x_train))
w=np.matmul(x_d,y_train)

yp=np.matmul(x_test,w)

plot_x=list(y_test)
plot_y=list(yp)

plt.scatter(plot_x, plot_y,color= "green",  marker= "*",s=100)
plt.plot([0,50],[0,50], color='black', linewidth=1, label='learning algorithm')
plt.ylabel('predicted price')
plt.xlabel('actual price')
plt.title('predicted price vs actual price')
plt.show()
