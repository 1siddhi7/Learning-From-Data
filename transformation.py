#radius=5
import matplotlib.pyplot as plt
import numpy as np
import random

#generating random data
x_in = list(np.random.uniform(-5,5,60))
y_in=[]
for i in range(60):
    y_in.append(np.random.uniform(-((25 - (x_in[i]) ** 2) ** 0.5),(25 - (x_in[i]) ** 2) ** 0.5))
o_in=[1]*60

x_out1 = list(np.random.uniform(-5,5,30))
y_out1=[]
for i in range(15):
    y_out1.append(np.random.uniform((25-(x_out1[i])**2)**0.5,10))
for i in range(15,30):
    y_out1.append(np.random.uniform(-10,-(25 - (x_out1[i]) ** 2) ** 0.5))

y_out2 = list(np.random.uniform(-5,5,30))
x_out2=[]
for i in range(15):
    x_out2.append(np.random.uniform((25-(y_out2[i])**2)**0.5,10))
for i in range(15,30):
    x_out2.append(np.random.uniform(-10,-(25 - (y_out2[i]) ** 2) ** 0.5))

x_out=x_out1+x_out2
y_out=y_out1+y_out2
o_out=[-1]*60

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plt.plot(x_in, y_in, color='green', linewidth = 0, marker='o', markerfacecolor='green', markersize=3,label="output : 1")
plt.plot(x_out, y_out, color='blue', linewidth = 0, marker='o', markerfacecolor='blue', markersize=3,label = "output : -1")
plt.legend()
centreCircle = plt.Circle((0,0),5,color="red",fill=False)
ax.add_patch(centreCircle)
plt.show()

data=[]
for i in range(60):
    data.insert(i,[x_in[i],y_in[i],o_in[i]])
i=60
for j in range(60):
    data.insert(i,[x_out[j],y_out[j],o_out[j]])
    i=i+1

random.shuffle(data)

transformed_data=[]

for i in range(120):
    x=data[i][0]**2
    y=data[i][1]**2
    transformed_data.insert(i,[1,x,y,data[i][2]])

x_green=[]
y_green=[]
x_blue=[]
y_blue=[]
for i in range(120):
    if(transformed_data[i][3]==1):
        x_green.append(transformed_data[i][1])
        y_green.append(transformed_data[i][2])
    elif (transformed_data[i][3] == -1):
        x_blue.append(transformed_data[i][1])
        y_blue.append(transformed_data[i][2])

plt.plot(x_green, y_green, color='green', linewidth = 0, marker='o', markerfacecolor='green', markersize=3,label="output : 1")
plt.plot(x_blue, y_blue, color='blue', linewidth = 0, marker='o', markerfacecolor='blue', markersize=3,label="output : -1")
plt.legend()
plt.show()

#using 80% of the data for training
cut_point = int(len(transformed_data) * 0.8)
train_data = transformed_data[:cut_point]

#using remaining 20% of the data for testing
test_data = transformed_data[cut_point:]


#applying perceptron learning algorithm on transforming_data
def sign(x):
    if x>=0:
        return 1
    if x<0:
        return -1

def misclassified(data,weights):
    flag=0
    for i in range(len(data)):
        x= np.array(data[i][:-1])
        y= data[i][-1]
        if(sign(np.matmul(weights.T,x))!=y):
            flag=1
            break
    return flag

weights=np.array([0,0,0])
iteration=0

while (misclassified(train_data, weights)):
    for i in range(len(train_data)):
        x = np.array(train_data[i][:-1])
        yn = train_data[i][-1]
        flag = 0
        if (sign(np.matmul(weights.T, x)) != yn):
            flag = 1

            iteration = iteration + 1

            weights = weights + yn * x

        if (flag == 1):
            break

for i in range(len(test_data)):
    xt = np.array(test_data[i][:-1])
    print(sign(np.matmul(weights.T,xt)),test_data[i][-1])


x_black=[]
y_black=[]
x_red=[]
y_red=[]

for i in range(len(test_data)):
    if(test_data[i][3] == sign(np.matmul(weights.T,np.array(test_data[i][:-1])))):
        x_black.append(test_data[i][1])
        y_black.append(test_data[i][2])
    elif (test_data[i][3] != sign(np.matmul(weights.T,np.array(test_data[i][:-1])))):
        x_red.append(test_data[i][1])
        y_red.append(test_data[i][2])

#plotting misclassified points in red colour and classified points in black colour
plt.plot(x_black, y_black, color='black', linewidth = 0, marker='o', markerfacecolor='black', markersize=3,label = "Correctly classified points")
plt.plot(x_red, y_red, color='red', linewidth = 0, marker='o', markerfacecolor='red', markersize=3,label= "Incorrectly classified points")
plt.legend()
plt.show()