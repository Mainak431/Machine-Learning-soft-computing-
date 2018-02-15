import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#supressing the scientific output
np.set_printoptions(suppress=True)
data = np.loadtxt("ex1data2.txt",dtype=np.float64,delimiter=",")
X = data[:25,0:2]
Y = data[0:25,-1:]
print(X)
print(Y)

m,n =X.shape
for i in range (m) :
    X2 = data[: 25,0:2]*data[:25,0:2]
    X3 = data[: 25,0:1]*data[:25,1:2]
print(X2)
print(X3)
Theta = np.random.rand(1,6)
n = n+4;
X_bias = np.ones((m,n))
X_bias[::,1:3] = X
X_bias[::,3:5] = X2
X_bias[::,-1:] = X3
print(X_bias)
mean_size = np.mean(X_bias[::,1:2])
mean_bedroom = np.mean(X_bias[::,2:3])
size_std = np.std(X_bias[::,1:2])
bedroom_std = np.std(X_bias[::,2:3])
mean_sizesq = np.mean(X_bias[::,3:4])
sizesq_std = np.std(X_bias[::,3:4])
print(Theta)
mean_bedroomsq = np.mean(X_bias[::,4:5])
bedroomsq_std = np.std(X_bias[::,4:5])
mean_both = np.mean(X_bias[::,-1: ])
both_std = np.std(X_bias[::,-1: ])
X_bias[::,1:2] = (X_bias[::,1:2] - mean_size)/ (size_std)
X_bias[::,2:3] = (X_bias[::,2:3] - mean_bedroom)/ (bedroom_std)
X_bias[::,3:4] = (X_bias[::,3:4] - mean_sizesq)/ (sizesq_std)
X_bias[::,4:5] = (X_bias[::,4:5] - mean_bedroomsq)/ (bedroomsq_std)
X_bias[::,-1:] = (X_bias[::,-1:] - mean_both)/ (both_std)
print(X_bias[:,::])
Xtrain = data[25:47,0:2]
Ytrain = data[25:47,-1:]
for i in range (22) :
    Xtrain2 = data[25: ,0:2]*data[25:,0:2]
    Xtrain3 = data[ 25:,0:1]*data[25:,1:2]
Xtrain_bias = np.ones((22,n))
Xtrain_bias[::,1:3] = Xtrain
Xtrain_bias[::,3:5] = Xtrain2
Xtrain_bias[::,-1:] = Xtrain3
Mintheta = np.ones((1,6))
MinCost=np.zeros((1,1))
pertheta = np.ones((1,6))
pertheta[0,0] = Theta[0,0]
pertheta[0,1] = Theta[0,1]
pertheta[0,2] = Theta[0,2]
pertheta[0,3] = Theta[0,3]
pertheta[0,4] = Theta[0,4]
pertheta[0,5] = Theta[0,5]
minlamda = np.zeros((1,1))
def cost(X_bias,Y,Theta,lamda):
    np.seterr(over='raise')
    hypothesis = X_bias.dot(Theta.transpose())
    return (1/(2.0*m))*lamda*((np.square(hypothesis-Y)).sum(axis=0))


def gradientDescent(X_bias,Y,Theta,iterations,alpha,lamda):
    count = 1
    gradientDescent.c += 1
    cost_logi = np.array([])
    Theta = Theta*(1-lamda*alpha/m)
    while(count <= iterations):
        hypothesis = X_bias.dot(Theta.transpose())
        temp0 = Theta[0,0] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,0:1])).sum(axis=0)
        temp1 = Theta[0,1] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,1:2])).sum(axis=0)
        temp2 = Theta[0,2] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,2:3])).sum(axis=0)
        temp3 = Theta[0, 3] - alpha * (1.0 / m) * ((hypothesis - Y) * (X_bias[::, 3:4])).sum(axis=0)
        temp4 = Theta[0, 4] - alpha * (1.0 / m) * ((hypothesis - Y) * (X_bias[::, 4:5])).sum(axis=0)
        temp5 = Theta[0, 5] - alpha * (1.0 / m) * ((hypothesis - Y) * (X_bias[::, -1:])).sum(axis=0)
        Theta[0,0] = temp0
        Theta[0,1] = temp1
        Theta[0,2] = temp2
        Theta[0,3] = temp3
        Theta[0,4] = temp4
        Theta[0,5] = temp5
        costi = +cost(X_bias,Y,Theta,lamda)
        count = count + 1
        cost_logi = np.append(cost_logi, cost(X_bias, Y, Theta,lamda))
    if gradientDescent.c == 1 :
        Mintheta[0,0] = Theta[0,0]
        Mintheta[0,1] = Theta[0,1]
        Mintheta[0,2] = Theta[0,2]
        Mintheta[0,3] = Theta[0,3]
        Mintheta[0,4] = Theta[0,4]
        Mintheta[0,5] = Theta[0,5]
        MinCost[0,0] = costi
        minlamda[0,0]=lamda
    elif costi < MinCost[0,0] :
        Mintheta[0, 0] = Theta[0, 0]
        Mintheta[0, 1] = Theta[0, 1]
        Mintheta[0, 2] = Theta[0, 2]
        Mintheta[0, 3] = Theta[0, 3]
        Mintheta[0, 4] = Theta[0, 4]
        Mintheta[0, 5] = Theta[0, 5]
        MinCost[0,0] = costi
        minlamda[0,0] = lamda
    """plt.plot(np.linspace(1,iterations,iterations,endpoint=True),cost_logi)
    plt.title("Iteration vs Cost graph ")
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost of Theta")
    plt.show()"""
    return Theta

gradientDescent.c = 0
alpha = 0.3
iterations = m
lamda = 0.2
while lamda < 1000000 :
    Theta = gradientDescent(X_bias,Y,pertheta,iterations,alpha,lamda)
    lamda = lamda*2
i = 1;
Theta[0, 0] = Mintheta[0, 0]
Theta[0, 1] = Mintheta[0, 1]
Theta[0, 2] = Mintheta[0, 2]
Theta[0, 3] = Mintheta[0, 3]
Theta[0, 4] = Mintheta[0, 4]
Theta[0, 5] = Mintheta[0, 5]
mini = 0
print(minlamda)
print(Theta)
Xtrain_predict = Xtrain_bias
Xtrain_predict[:,1:2] = (Xtrain_predict[:,1:2] - mean_size) / (size_std)
Xtrain_predict[:,2:3] = (Xtrain_predict[:,2:3] - mean_bedroom) / (bedroom_std)
Xtrain_predict[:,3:4] = (Xtrain_predict[:,3:4] - mean_sizesq) / (sizesq_std)
Xtrain_predict[:,4:5] = (Xtrain_predict[:,4:5]) - mean_bedroomsq / (bedroomsq_std)
Xtrain_predict[:,-1:] = (Xtrain_predict[:,-1:]) - mean_both / (both_std)
T1 = np.ones((1,6))
T1[0,0]=Theta[0,0]
T1[0,1]=Theta[0,1]
T1[0,2]=Theta[0,2]
T1[0,3]=Theta[0,3]
T1[0,4]=Theta[0,4]
T1[0,5]=Theta[0,5]
i = 0
T = np.ones((1,6))
for j in range(6):
    Theta[0, 0] = T1[0, 0]
    Theta[0, 1] = T1[0, 1]
    Theta[0, 2] = T1[0, 2]
    Theta[0, 3] = T1[0, 3]
    Theta[0, 4] = T1[0, 4]
    Theta[0, 5] = T1[0, 5]
    for k in range(j):
        Theta[0,5-k] = 0.002
    k = 0
    sumi = 0
    for k in range (22):
        sumi += np.square(Xtrain_predict[k:k+1, : ].dot(Theta.transpose())- Ytrain[k:k+1, ])
    if i == 0:
        mini = sumi
        T[0,0]=Theta[0,0]
        T[0,1]=Theta[0,1]
        T[0,2]=Theta[0,2]
        T[0,3]=Theta[0,3]
        T[0,4]=Theta[0,4]
        T[0,5]=Theta[0,5]
    elif sumi < mini:
        mini = sumi
        T[0, 0] = Theta[0, 0]
        T[0, 1] = Theta[0, 1]
        T[0, 2] = Theta[0, 2]
        T[0, 3] = Theta[0, 3]
        T[0, 4] = Theta[0, 4]
        T[0, 5] = Theta[0, 5]
    i=i+1
print (T)
X_predict = np.array([1.0,0,0,0.0,0.0,0.0])
#feature scaling the data first
X_predict[1]= input("Enter Floor area")
X_predict[2]= input("Enter No of Bedrooms")
a = X_predict[1]
b = X_predict[2]
X_predict[3] = a*a
X_predict[4]= b*b
X_predict[5] = a*b
X_predict[1] = (X_predict[1] - mean_size)/ (size_std)
X_predict[2] = (X_predict[2]- mean_bedroom)/ (bedroom_std)
X_predict[3] = (X_predict[3] - mean_sizesq)/(sizesq_std)
X_predict[4] = (X_predict[4]) - mean_bedroomsq/(bedroomsq_std)
X_predict[5] = (X_predict[5]) - mean_both/(both_std)
hypothesis = X_predict.dot(T.transpose())
print ("Cost of house with {0} sq ft and {1} bedroom is {2}$".format(a,b,hypothesis))
print("Prediction Percentage")
while(1):
    tol = np.zeros((1,1))
    tol[0,0]=input("Enter tolerance is $")
    co = 0;
    for i in range(47):
        Y = data[i:i+1, -1:]
        X_predict[1]=data[i:i+1,0:1]
        X_predict[2]=data[i:i+1,1:2]
        X_predict[3] = a * a
        X_predict[4] = b * b
        X_predict[5] = a * b
        X_predict[1] = (X_predict[1] - mean_size) / (size_std)
        X_predict[2] = (X_predict[2] - mean_bedroom) / (bedroom_std)
        X_predict[3] = (X_predict[3] - mean_sizesq) / (sizesq_std)
        X_predict[4] = (X_predict[4]) - mean_bedroomsq / (bedroomsq_std)
        X_predict[5] = (X_predict[5]) - mean_both / (both_std)
        hypothesis = X_predict.dot(T.transpose())
        dif = np.abs(Y-hypothesis)
        if dif < tol:
            co = co+1
    res = co/0.47
    print(res)