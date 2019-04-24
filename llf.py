n=int(input("Enter No. of processes: "))
print("Enter Execution time, Period and Deadline of Processes")
e=[]
p=[]
d=[]
l=[1]*n
parts=[0]*n
for i in range(n):
    exe,per,dead=input().split()
    e.append(int(exe))
    p.append(int(per))
    d.append(int(dead))

e_inter=e
from math import gcd
lcm = p[0]
for i in p[1:]:
  lcm = int(lcm*i/gcd(lcm, i))


time = lcm

for i in range(7):
    laxity=[0]*n
    for j in range(n):
        laxity[j]=d[j]-((i%p[j])+e_inter[j])
    print(laxity)

    for j in range(n):
        if(l[j]==1):
            pr=j
        
    for j in range(n):
        if(laxity[j]<=laxity[pr] and l[j]==1):
            pr=j
        
    print("From",i,"to",i+1,": process",pr+1)
    e_inter[pr]=e_inter[pr]-1

    print(e_inter)
    for j in range(n):
        if(e_inter[j])==0:
            l[j]=0
        if(i % p[j]==0):
            e_inter[j]=e[j]
            l[j]=1
    
        



    
    

      
