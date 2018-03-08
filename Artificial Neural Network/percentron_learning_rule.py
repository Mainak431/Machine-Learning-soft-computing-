import numpy
w0 = 0.5
w1 = -0.5
w2 = -0.5


def sigmoid(z):
    return 1/(1+numpy.exp(-z))
x = ([0,0],
     [0,1],
     [1,0],
     [1,1])
#and gate expected output
#y = ([0, 0, 0, 1])
# or gate expected output
#y = ([0, 1, 1 ,1])
#nor gate expected output
y = ([1, 0, 0 ,0])
#nand gate expected output
#y = ([1, 1, 1 ,0])
alpha = 0.1
while (1):
    count = 0;
    res = w0 + w1*x[0][0]+w2*x[0][1]
    error1 = y[0] - sigmoid(res)
    if abs(error1) > 0.001:
        w0 = w0 + alpha*1*error1
        w1 = w1 + alpha*x[0][0]*error1
        w2 = w2 + alpha*x[0][1]*error1
        count = count + 1
    res = w0 + w1 * x[1][0] + w2 * x[1][1]
    error2 = y[1] - sigmoid(res)
    if abs(error2) > 0.001:
        w0 = w0 + alpha * 1 * error2
        w1 = w1 + alpha * x[1][0] * error2
        w2 = w2 + alpha * x[1][1] * error2
        count = count + 1
    res = w0 + w1 * x[2][0] + w2 * x[2][1]
    error3 = y[2] - sigmoid(res)
    if abs(error3) > 0.001:
        w0 = w0 + alpha * 1 * error3
        w1 = w1 + alpha * x[2][0] * error3
        w2 = w2 + alpha * x[2][1] * error3
        count = count + 1
    res = w0 + w1 * x[3][0] + w2 * x[3][1]
    error4 = y[3] - sigmoid(res)
    if abs(error4) > 0.001:
        w0 = w0 + alpha * 1 * error4
        w1 = w1 + alpha * x[3][0] * error4
        w2 = w2 + alpha * x[3][1] * error4
        count = count + 1
    if (count == 0):
        break
weight = [w0,w1,w2]
print(weight)
res = w0 + w1*x[1][0]+w2*x[1][1]
print(sigmoid(res))