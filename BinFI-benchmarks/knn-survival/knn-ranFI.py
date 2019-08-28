#!/usr/bin/python
'''
A nearest neighbor learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
import TensorFI as ti
from TensorFI import faultTypes as ft
import numpy, pandas
import preprocessing
import datetime
import numpy as np
import sys
import os 

######
data = pandas.read_csv("./survive.csv")
data = preprocessing.cleanDataForClassification(data, "class")
 
labels = []
for d in data['class']:
    if int(d) == 1:
	labels.append([0,1])
    if int(d) == 2:
	labels.append([1,0])
labels = pandas.DataFrame(labels).values
######
 
batch_xs = data.drop("class",axis=1).values
batch_ys = labels
Xtr = batch_xs[:150]
Ytr = batch_ys[:150]
Xtest = batch_xs[150:]
Ytest = batch_ys[150:]

# tf Graph Input
xtr = tf.placeholder("float", [None, 3]) # 3 features in survive dataset
xte = tf.placeholder("float", [3])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)


neg = tf.negative(xte)
ad = tf.add(xtr, neg)
ab = tf.abs(ad)
distance = tf.reduce_sum(ab, reduction_indices=1)


accuracy = 0.

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)


    # we use the inputs that can be correctly identified by the model for FI
    Xte = Xtest[:30]
    Yte = Ytest[:30]
    wrong = []
    for sampleIndex in range(Xte.shape[0]): 
        nn_index = pred.eval({xtr: Xtr, xte: Xte[sampleIndex , :]})
        if np.argmax(Ytr[nn_index]) != np.argmax(Yte[sampleIndex]):
           wrong.append(sampleIndex)
    Xte = np.delete(Xte, wrong, axis=0) 
    Yte = np.delete(Yte, wrong, axis=0)  
    # now the inputs in test set can all be correctly identified by the models


    # Add the fault injection code here to instrument the graph 
    fi = ti.TensorFI(sess, name = "NearestNeighbor", logLevel = 50, disableInjections = True)
    fi.turnOnInjections()
    # loop over test data

    # save FI results into file, "eachRes" saves each FI result, "resFile" saves SDC rate
    resFile = open('sknn-ranFI.csv', "a")
    eachRes = open('sknn-eachRanFI.csv', "a")

    totalFI = 4650 # specify the number of random FI trials you want to conduct
    for sampleIndex in range(10):
 
        sdcCount = 0.

        for j in range(totalFI):
            nn_index = sess.run([pred], feed_dict={xtr: Xtr, xte: Xte[sampleIndex, :]}) 
 
            if(np.argmax(Ytr[nn_index]) != np.argmax(Yte[sampleIndex])):
                # FI results in SDC
                sdcCount += 1
                eachRes.write(`0` + ",")
            else:
                eachRes.write(`1` + ",")

            print(j, sampleIndex)

        eachRes.write("\n")
        print("sdc:", sdcCount/totalFI, sdcCount)
        resFile.write(`sdcCount/totalFI` + "," + `sdcCount` + "\n")
  


