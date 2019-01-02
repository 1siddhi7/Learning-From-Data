import numpy as np
import random
import matplotlib.pyplot as plt

bin_size=3000
bin=list(np.random.randint(2, size=bin_size))
countb_0=bin.count(0)
countb_1=bin.count(1)
mu = countb_0/(countb_0+countb_1)

print("mu is",mu)

l=list(range(40,bin_size+1,40))

plot = []
for i in range(40,bin_size+1,40):

    lnu = []
    for j in range(100):
        sample=[]
        for k in range(i):
            sample.append(random.choice(bin))
        counts_0 = sample.count(0)
        counts_1 = sample.count(1)
        nu = counts_0 / (counts_0 + counts_1)
        lnu.append(nu)

    plot.append(lnu)

plot_final=[]
for i in range(40,bin_size+1,40):
    nu_avg=sum(plot[l.index(i)])/100
    plot_final.append(nu_avg)


std_deviation=[]
for i in range(40,bin_size+1,40):
    std_dev=0
    for j in range(100):
        std_dev = std_dev + (plot_final[l.index(i)]-(plot[l.index(i)][j]))**2
    std_deviation.append((std_dev/100)**0.5)

x = [item for item in l]
y = [item for item in plot_final]


plt.errorbar(x,y,yerr=std_deviation,ecolor='black',marker='*',markersize=5,capsize=3,label='avg_nu')
plt.plot([0,bin_size],[mu,mu],color='red',linewidth=2,label='mu')
plt.xlabel('sample size')
plt.legend()
plt.show()

