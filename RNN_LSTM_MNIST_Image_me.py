import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import time
import random
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


# Initializations

# constants
learning_rate = 0.003
epochs = 10000
batch_size = 128
display_step = 200

# model constants
n_hidden = 128
n_inputs = 28
timesteps = 28
test_len = 128
num_classes = 10


# placeholders
X = tf.placeholder(tf.float32,[None,timesteps,n_inputs])
Y = tf.placeholder(tf.float32,[None,num_classes])


# weights and biases for output layer

W = tf.Variable(tf.random_normal([n_hidden,num_classes]))
B = tf.Variable(tf.random_normal([num_classes]))

# Model Creation

def RNN_model(x,W,B):
    x = tf.unstack(x,timesteps,1)
    lstm_cells = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)],state_is_tuple=True)
    outputs,states = rnn.static_rnn(lstm_cells,x,dtype=tf.float32)
    return tf.matmul(outputs[-1],W)+B



Ylogits = RNN_model(X,W,B)
Ypred = tf.nn.softmax(Ylogits)


# Success metrics = accuracy+loss+optimizer

is_correct = tf.equal(tf.argmax(Ypred,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(loss)


# For saving and restoring model
saver = tf.train.Saver()

# Actual Training

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    steps = 0
    total_acc = 0
    total_loss = 0
    for epoch in range(0,epochs):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size,timesteps,n_inputs])
        sess.run(train_step,feed_dict={X:batch_x,Y:batch_y})
        a,l = sess.run([accuracy,loss],feed_dict={X:batch_x,Y:batch_y})

        total_loss += l
        total_acc += a
        steps+=1
        if steps % display_step == 0:
            print("EPOCH = {}+++> : Total loss : {} , Total accuracy : {}.".format(steps,total_loss/display_step,(total_acc/display_step)*100))
            total_loss = 0
            total_acc = 0 

    saver.save(sess,"mymodels/model-mnist-rnn.ckpt")
    test_x = np.array(mnist.test.images[:test_len]).reshape([-1,timesteps,n_inputs])
    test_y = np.array(mnist.test.labels[:test_len]).reshape([-1,num_classes])
    acc,l = sess.run([accuracy,loss],feed_dict={X:test_x,Y:test_y})
    print("Test : Total loss : {} , Total accuracy : {}.".format(l,acc*100))	



