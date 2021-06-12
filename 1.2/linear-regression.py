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

# plot loss values
plt.plot(loss_values, 'ro')
plt.show()