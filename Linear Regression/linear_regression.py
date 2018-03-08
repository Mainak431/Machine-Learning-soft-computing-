import tensorflow as tf
import pandas as pd
import turtle

x = tf.placeholder(tf.float32)
w = tf.Variable([[0.01],[0.01]],dtype=tf.float32)
b = tf.Variable([0.01],dtype=tf.float32)
y = tf.matmul(x,w) + b
yg = tf.placeholder(tf.float32)
loss = tf.reduce_mean(tf.square(y-yg))
optimizer = tf.train.GradientDescentOptimizer(0.00000001)
train = optimizer.minimize(loss)
x_train =pd.DataFrame([[2014,3],[1600,3],[2400,3],[1416,2],[3000,4]])
y_train =pd.DataFrame([400,330,369,232,540])
gVar = tf.global_variables_initializer()
ses = tf.Session()
ses.run(gVar)
for i in range(10000):
    for j in range(0,5):
        X = x_train.iloc[j:j+1,0:2]
        Y = y_train.iloc[j]
        ses.run(train, {x: X, yg: Y})
x2check = float(input("Input no of bedrooms"))
x1check = float(input("Input floor area"))
xfinal = pd.DataFrame([[x1check,x2check]])
print(ses.run(y,{x:xfinal}))
