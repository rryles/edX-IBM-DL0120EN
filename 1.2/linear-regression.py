import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress warning that it can't use CUDA

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

df = pd.read_csv("FuelConsumption.csv")

train_x = np.asanyarray(df[['ENGINESIZE']])
train_y = np.asanyarray(df[['CO2EMISSIONS']])

# Model parameters
a = tf.Variable(30.0)
b = tf.Variable(1.0)

# Model
def h(x):
   y = a*x + b
   return y

# Loss function that we will minimise
def loss_object(y, train_y):
    return tf.reduce_mean(tf.square(y - train_y))

learning_rate = 0.01
train_data = []
loss_values = []

training_epochs = 200

# Train Model
for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        # calculate loss
        y_predicted = h(train_x)
        loss_value = loss_object(train_y, y_predicted)
        loss_values.append(loss_value)

        # get gradients
        gradients = tape.gradient(loss_value, [a, b])

        # adjust weights
        a.assign_sub(gradients[0] * learning_rate)
        b.assign_sub(gradients[1] * learning_rate)
        if epoch % 5 == 0:
            train_data.append([a, b])

# Display final model vs training data
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(train_x)
    line = plt.plot(train_x, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(train_x, train_y, 'ro')
green_line = mpatches.Patch(color='red', label='Data Points')

plt.legend(handles=[green_line])

plt.show()