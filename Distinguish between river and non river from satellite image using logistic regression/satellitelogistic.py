import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


def sigmoid(Z):
    out = 1 / (1 + np.exp(-Z))
    return out


face1 = misc.imread('1.gif')
"""plt.imshow(face1)
plt.show()"""
print(face1)
face2 = misc.imread('2.gif')
"""plt.imshow(face2)
plt.show()"""
print(face2)
face3 = misc.imread('3.gif')
"""plt.imshow(face3)
plt.show()"""
print(face3)
face4 = misc.imread('4.gif')
"""plt.imshow(face4)
plt.show()"""
print(face4)
data = np.loadtxt("river.txt",dtype=np.int64,delimiter=",")
X = data[:,0:1]
Y = data[:,1:2]
data2 = np.loadtxt("Nonriver.txt",dtype=np.int64,delimiter=",")
X2 = data2[:,0:1]
Y2 = data2[:,1:2]
m,n =X.shape
m2,n2 = X2.shape
pr1 = np.ones((m,1))
for i in range(m):
    pr1[i] = face1[X[i],Y[i],1]
#print(pr1)
pn1 = np.ones((m2,1))
for i in range(m2):
    pn1[i] = face1[X2[i],Y2[i],1]
#print(pn1)
pr2 = np.ones((m,1))
for i in range(m):
    pr2[i] = face2[X[i],Y[i],1]
#print(pr2)
pn2 = np.ones((m2,1))
for i in range(m2):
    pn2[i] = face2[X2[i],Y2[i],1]
#print(pn2)
pr3 = np.ones((m,1))
for i in range(m):
    pr3[i] = face3[X[i],Y[i],1]
#print(pr3)
pn3 = np.ones((m,1))
for i in range(m2):
    pn3[i] = face3[X2[i],Y2[i],1]
#print(pn3)
pr4 = np.ones((m,1))
for i in range(m):
    pr4[i] = face4[X[i],Y[i],1]
#print(pr4)
pn4 = np.ones((m2,1))
for i in range(m2):
    pn4[i] = face4[X2[i],Y2[i],1]
#print(pn4)
m,n = pn1.shape
print(m)
print(n)
d1 = np.ones((2*m,n))
d2 = np.ones((2*m,n))
d3 = np.ones((2*m,n))
d4 = np.ones((2*m,n))
d1[0:m,0:1] = pr1
d1[m:,0:1] = pn1
d2[0:m,0:1]= pr2
d2[m: , :] = pn2
d3[0:m,:] = pr3
d3[m:,:] = pn3
d4[:m,:] =pr4
d4[m:,:]=pn4
print(d4)

mean1 = np.mean(d1)
mean2 = np.mean(d2)
mean3 = np.mean(d3)
mean4 = np.mean(d4)

std1 = np.std(d1)
std2 = np.std(d2)
std3 = np.std(d3)
std4 = np.std(d4)

d1 = (d1 - mean1)/std1
d2 = (d2 - mean2)/std2
d3 = (d3 - mean3)/std3
d4 = (d4 - mean4)/std4


X_bias = np.ones((2*m,n*4+1))
X_bias[:, 1:2] = d1
X_bias[:,2:3] = d2
X_bias[:,3:4] = d3
X_bias[:,4:] = d4
print(X_bias)

W = np.zeros((5,1))
iteration = 10000
alpha = 0.3

Y = np.ones((2*m,1))
Y[:m,:] = data[:,-1:]
Y[m:] = data2[:,-1:]
print(Y)
for i in range(iteration) :
    hypothesis = sigmoid(np.dot(X_bias,W))
    J = sum( - (np.dot(Y.transpose(),np.log(hypothesis)) +( np.dot((1 - Y).transpose(), np.log(1 - hypothesis)))))
    inc = np.dot(X_bias.transpose(), (Y - hypothesis))
    W = (W )+ ((alpha/2*m)*inc);

test = misc.imread('4.gif')
m,n,p = test.shape
outimg = np.zeros([512,512,3],dtype=np.uint8)
outimg.fill(255)
x_predict = np.zeros((m,n))
for i in range(m):
    for j in range(n):
        x_predict[i, j] = np.dot([1, (((test[i, j,1]) - mean1)/std1), ((test[i, j,1]) - mean2)/std2,(((test[i, j,1]) - mean3)/std3),(((test[i, j,1]) - mean4)/std4)],W)
        x_predict[i,j] = sigmoid(x_predict[i,j])
        if x_predict[i,j] >= 0.5:
            outimg[i,j] = 255
        else :
            outimg[i,j] = 0
plt.imshow(outimg);
plt.show()