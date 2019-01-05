import numpy as np
import random
import matplotlib.pyplot as plt

d = { "Iris-setosa":1,"Iris-versicolor":-1,"Iris-virginica":-1}
features = []
labels = []
data = []

with open("iris.data.txt", 'r') as f:
    for line in f:

        if not line.strip():
            continue
        sl, sw, pl, pw, species = line.strip().split(',')

        labels.append(d[species])
        l = [1, float(sl), float(sw), float(pl), float(pw)]
        features.append(l)
        data.append([l, d[species]])

print(data)
random.shuffle(data)

# using 80% of the data for training
cut_point = int(len(data) * 0.8)
train_data = data[:cut_point]

# using remaining 20% of the data for testing
test_data = data[cut_point:]


def sign(x):
    if x >= 0:
        return 1
    if x < 0:
        return -1


weights=np.array([0,0,0,0,0])  #initializing the weights vector


error = []
iteration = 0
e = 0
l_pocket=[]

for j in range(len(train_data)):
    x = np.array(train_data[j][0])
    yn = train_data[j][1]
    yn_predicted = sign(np.matmul(weights.T, x))
    e = e + (yn - yn_predicted) ** 2
e = e ** 0.5  # the initial root mean square error with weights vector as 0 vector

error.append([iteration, e])
pocket=[weights,e]#At each iteration, we will pocket the weights vector which gives minimum error
l_pocket.append(pocket)

# training using pocket learning algorithm; terminating the training at 5th iteration
for k in range(6):
    for i in range(len(train_data)):
        x = np.array(train_data[i][0])
        yn = train_data[i][1]
        flag = 0
        if (sign(np.matmul(weights.T, x)) != yn):
            flag = 1

            iteration = iteration + 1

            weights = weights + yn * x

        if (flag == 1):
            e = 0
            for j in range(len(train_data)):
                x = np.array(train_data[j][0])
                yn = train_data[j][1]
                yn_predicted = sign(np.matmul(weights.T, x))
                e = e + (yn - yn_predicted) ** 2
            e = e ** 0.5
            error.append([iteration, e])

        if(e<pocket[1]):
            pocket=[weights,e]

        if (flag == 1):
            break
    l_pocket.append(pocket)


weights=pocket[0]
e=pocket[1]

# testing the model using test_data
for i in range(len(test_data)):
    xt = np.array(test_data[i][0])
    print(sign(np.matmul(weights.T, xt)),
          test_data[i][1])  # printing the predicted output and actual output for each data item in test_data

# plotting a graph of the root mean square error and the pocketed error at each iteration vs the no. of iterations during training
x = []
y = []
for i in error:
    y.append(i[1])
for i in error:
    x.append(i[0])

pock =[item[1] for item in l_pocket]

plt.plot(x, y, color='green', linewidth=2, marker='o', markerfacecolor='blue', markersize=8,label='learning algorithm')
plt.plot([0,1,2,3,4,5,6],pock,color='red',linewidth=2,label='pocketed error')

plt.ylabel('error')
plt.xlabel('no of iterations')
plt.title('root mean square error vs no. of iterations')
plt.legend()
plt.show()
