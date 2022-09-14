import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm


Ndata = 500 # number of data points
dim = 6 # range of data points

# plot configuration
x = np.linspace(-dim, dim, 50)
y = np.linspace(-dim, dim, 50)
X_, Y_ = np.meshgrid(x, y)

# Gerate data
def XOR(a,b):
  return ( a > 0 and b > 0 ) or (a < 0 and b < 0)

data = np.array([[0,0]])
labels = np.array([True])

for _ in range(Ndata):
  x = random.uniform(-dim,dim)
  y = random.uniform(-dim,dim)
  eti = XOR(x,y)
  point = np.array([x,y])
  data = np.append(data,[point],axis=0)
  labels = np.append(labels,[eti],axis=0)

# Show Data
X = data[:,0]
Y = data[:,1]

colors = ["#ff8000",  # Orange
           "#0000ff"   # Blue
          ]
plt.scatter(X, Y,s=8,c=np.take(colors,labels))
plt.show()


# Activation Funtions

def tanh(x):
    return np.tanh(x)

def tanhD(x):
    return 1-(tanh(x)**2)

def sig(x):
    return 1/(1+np.exp(-x))

def sigD(x):
    return sig(x)*(1-sig(x))

# NeralnetWork


n1 = [random.uniform(-1,1),random.uniform(-1,1)]
b1 = random.uniform(-1,1)
n2 = [random.uniform(-1,1),random.uniform(-1,1)]
b2 = random.uniform(-1,1)
n3 = [random.uniform(-1,1),random.uniform(-1,1)]
b3 = random.uniform(-1,1)

print(n1)

def model(x,y):
     return sig(tanh(x*n1[0]+y*n1[1]+b1)*n3[0]+tanh(x*n2[0]+y*n2[1]+b2)*n3[1]+b3)


alfa = 0.03
Error = 100
for _ in range(250):
 Error = 0
 for i in range(Ndata):

   In = data[i]
   label = labels[i]
   x = In[0]
   y = In[1]
   pred = model(x,y)


   #Compute Error
   error = (pred-label)**2

   Error+=error

   A = sigD(tanh(x*n1[0]+y*n1[1]+b1)*n3[0]+tanh(x*n2[0]+y*n2[1]+b2)*n3[1]+b3)
   n3[0] = n3[0] - alfa * 2 * (pred-label) * A * tanh(x*n1[0]+y*n1[1]+b1)
   n3[1] = n2[1] - alfa * 2 * (pred-label) * A * tanh(x*n2[0]+y*n2[1]+b2)
   b3 = b3 - alfa * 2 * (pred-label) * A * 1

   B = tanhD(x*n2[0]+y*n2[1]+b2)
   n2[0] = n2[0] - alfa * 2 * (pred-label) * A * n3[1] * B * x
   n2[1] = n2[1] - alfa * 2 * (pred-label) * A * n3[1] * B * y
   b2 = b2 - alfa * 2 * (pred-label) * A * n3[1]* B * 1

   C = tanhD(x*n1[0]+y*n1[1]+b1)
   n1[0] = n1[0] - alfa * 2 * (pred-label) * A * n3[0] * C * x
   n1[1] = n1[1] - alfa * 2 * (pred-label) * A * n3[0] * C * y
   b1 = b1 - alfa * 2 * (pred-label) * A * n3[0]* B * 1


 print(Error)




plt.contourf(X_, Y_, model(X_,Y_), 30 , cmap='Blues')
plt.scatter(X, Y,s=8,c=np.take(colors,labels))
plt.show()
