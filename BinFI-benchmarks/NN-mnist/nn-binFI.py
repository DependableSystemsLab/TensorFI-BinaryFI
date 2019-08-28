#!/usr/bin/python

""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import TensorFI as ti
mnist = input_data.read_data_sets("./mnistData/", one_hot=True)

import tensorflow as tf
import math
import sys
import numpy as np
import datetime

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1], seed = 10)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], seed = 10)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes], seed = 10))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], seed = 10)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], seed = 10)),
    'out': tf.Variable(tf.random_normal([num_classes], seed = 10))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#    layer_2 = tf.nn.dropout(layer_2, 0.2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model 
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
 
    num_steps = 10
    display_step = 1
    
    Xtr, Ytr = mnist.train.next_batch(1000, shuffle=False)
    
    for step in range(1, num_steps+1):
        batch_x = Xtr[(step-1)*100: step*100]
        batch_y = Ytr[(step-1)*100: step*100]
    	# Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
    print("Training Finished!")

    # Add the fault injection code here to instrument the graph
    fi = ti.TensorFI(sess, name = "Perceptron", logLevel = 50, disableInjections = True)
 
    # we use the inputs that can be correctly identified by the model for FI
    Xte = mnist.test.images[:2000]
    Yte = mnist.test.labels[:2000] 
    wrong = []
    for sampleIndex in range(Xte.shape[0]): 
        acy = accuracy.eval({X: Xte[sampleIndex:sampleIndex+1] , Y: Yte[sampleIndex:sampleIndex+1]})
        if(acy!=1):
           wrong.append(sampleIndex)
    Xte = np.delete(Xte, wrong, axis=0) 
    Yte = np.delete(Yte, wrong, axis=0)  
    # now the inputs in test set can all be correctly identified by the models

    # inputs to be injected
    indexs = [5, 64, 212, 313, 553, 610, 686, 697, 839, 857]  

    fi.turnOnInjections()
    # save FI results into file, "eachRes" saves each FI result, "resFile" saves SDC rate
    eachRes = open("mnn-binEach.csv", "a")
    resFile = open('mnn-binFI.csv', "a")
    for sampleIndex in indexs:
        fiTrial = 0
        # initiliaze for binary FI
        ti.faultTypes.initBinaryInjection() 
        while(ti.faultTypes.isKeepDoingFI):
            tx = Xte[sampleIndex:sampleIndex+1]
            ty = Yte[sampleIndex:sampleIndex+1] 
            acy = accuracy.eval({X: tx, Y: ty })
            # you need to feedback the FI result to guide the next FI for binary search
            if(acy == 1):
                # FI does not result in SDC
                ti.faultTypes.sdcFromLastFI = False 
            else:
                ti.faultTypes.sdcFromLastFI = True

            # if FI on the current data item, you might want to log the sdc bound for the bits of 0 or 1
            # (optional), if you just want to measure the SDC rate, you can access the variable of "ti.faultTypes.sdcRate"
            if(ti.faultTypes.isDoneForCurData):
                eachRes.write(`ti.faultTypes.sdc_bound_0` + "," + `ti.faultTypes.sdc_bound_1` + ",")
                # Important: initialize the binary FI for next data item.
                ti.faultTypes.initBinaryInjection(isFirstTime=False)

            fiTrial += 1
            print(sampleIndex, fiTrial)

        eachRes.write("\n")
        print("sdc", ti.faultTypes.sdcRate, "fi times: ", ti.faultTypes.fiTime)
        resFile.write(`ti.faultTypes.sdcRate` + "," + `ti.faultTypes.fiTime` + "\n")






