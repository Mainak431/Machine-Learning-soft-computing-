import numpy
import itertools
import random


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def RELU(x):
    return max(0, x)


n = 2
z = 2 ** n
y = numpy.zeros((z))
# generating input and output array
lst = list(map(list, itertools.product([0, 1], repeat=n)))
for i in range(z):
    count = 0
    for j in range(n):
        count += lst[i][j]
    if count % 2 == 0:
        y[i] = 0
    else:
        y[i] = 1
print(lst)
print(y)

layer1 = numpy.zeros((n, 1))
layer2 = numpy.zeros((2, 1))
# taking one hidden layer
# layer 3 is the final output single node
w = numpy.zeros((n * 2 + 2 * 1))
for i in range(n * 2 + 2 * 1):
    if i % 2 == 0:
        w[i] = 2.4 / n
    else:
        w[i] = -2.4 / n
alpha = 1
beta = 0.95
print(w)
y1 = 0
y2 = 0
y3 = 0
past_w = w
while (1):
    # setting up epoch
    e = 0
    for i in range(z):
        temp = numpy.zeros((n))
        for j in range(n):
            temp[j] = lst[i][j]
        weight1 = w[0:n]
        weight2 = w[n:2 * n]
        weight3 = w[2 * n:2 * n + 1]
        weight4 = w[-1:]
        y1 = 0.8
        y2 = -0.8
        y3 = 0.5
        # y1,y2,y3 bias to the network
        # randomly generating hidden layer and output layer
        caly1 = sigmoid(numpy.dot(numpy.transpose(weight1), temp) - y1)
        caly2 = sigmoid(numpy.dot(numpy.transpose(weight2), temp) - y2)
        caly = sigmoid(weight3 * caly1 + weight4 * caly2 - y3)
        error = y[i] - caly
        del5 = caly * (1 - caly) * error
        delw3 = caly1 * alpha * del5 + beta * (w[2 * n] - past_w[2 * n])
        delw4 = caly2 * alpha * del5 + beta * (w[2 * n + 1] - past_w[2 * n + 1])
        dely3 = alpha * (-1) * del5
        del3 = caly1 * (1 - caly1) * del5 * weight3
        del4 = caly2 * (1 - caly2) * del5 * weight4
        dely2 = alpha * (-1) * del4
        dely1 = alpha * (-1) * del3
        for m in range(2 * n):
            if m < n:
                w[m] = w[m] + alpha * lst[i][m % n] * del3 + beta * (w[m] - past_w[m])
            else:
                w[m] = w[m] + alpha * lst[i][m % n] * del4 + beta * (w[m] - past_w[m])
        w[2 * n] = weight3 + delw3
        w[-1] = weight4 + delw4
        y3 = y3 + dely3
        y2 = y2 + dely2
        y1 = y1 + dely1
        e += ((error) ** 2)
    print(e)
    if (e < 0.001):
        break
    past_w = w

print(w)
output1 = sigmoid(w[4] * sigmoid(w[0] * 0 + w[1] * 0 - y1) + w[5] * sigmoid(w[2] * 0 + w[3] * 0 - y2) - y3)
print(output1)
output1 = sigmoid(w[4] * sigmoid(w[0] * 0 + w[1] * 1 - y1) + w[5] * sigmoid(w[2] * 0 + w[3] * 1 - y2) - y3)
print(output1)
output1 = sigmoid(w[4] * sigmoid(w[0] * 1 + w[1] * 0 - y1) + w[5] * sigmoid(w[2] * 1 + w[3] * 0 - y2) - y3)
print(output1)
output1 = sigmoid(w[4] * sigmoid(w[0] * 1 + w[1] * 1 - y1) + w[5] * sigmoid(w[2] * 1 + w[3] * 1 - y2) - y3)
print(output1)
