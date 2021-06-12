import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress warning that it can't use CUDA

import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
iris_X = iris.data[:-1, :]
iris_Y = pd.get_dummies(iris.target[:-1]).values
trainX, testX, trainY, testY = train_test_split(iris_X, iris_Y, test_size=0.33, random_state=42)

numFeatures = trainX.shape[1]
numLabels = trainY.shape[1]

X = tf.Variable(np.identity(numFeatures), tf.TensorShape(numFeatures), dtype='float32')
yGold = tf.Variable(np.ones(numLabels), tf.TensorShape(numLabels), dtype='float32')

weights = tf.Variable(tf.random.normal([numFeatures,numLabels],
                                       mean=0.,
                                       stddev=0.01,
                                       name="weights"),dtype='float32')


bias = tf.Variable(tf.random.normal([1,numLabels],
                                    mean=0.,
                                    stddev=0.01,
                                    name="bias"))

# model
def logistic_regression(x):
    apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
    add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias") 
    activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")
    return activation_OP

# Number of Epochs in our training
numEpochs = 7000

# Defining our learning rate iterations (decay)
learningRate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.08,
                                          decay_steps=trainX.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)

#Defining our cost function - Squared Mean Error
loss_object = tf.keras.losses.MeanSquaredLogarithmicError()
optimizer = tf.keras.optimizers.SGD(learningRate)

# Accuracy metric.
def accuracy(y_pred, y_true):
# Predicted class is the index of the highest score in prediction vector (i.e. argmax).
    # print('y_pred : ',y_pred)
    # print('y_true : ',y_true)
    correct_prediction = tf.equal(tf.argmax(y_pred, -1), tf.argmax(y_true, -1))

    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimization process. 
def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = logistic_regression(x)
        loss = loss_object(pred, y)
    gradients = g.gradient(loss, [weights, bias])
    optimizer.apply_gradients(zip(gradients, [weights, bias]))

# Initialize reporting variables
display_step = 10
epoch_values = []
accuracy_values = []
loss_values = []
loss = 0
diff = 1

# Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .000001:
        print("change in loss %g; convergence."%diff)
        break
    else:
        # Run training step
        run_optimization(X, yGold)
        
        # Report occasional stats
        if i % display_step == 0:
            # Add epoch to epoch_values
            epoch_values.append(i)
            
            pred = logistic_regression(X)

            newLoss = loss_object(pred, yGold)
            # Add loss to live graphing variable
            loss_values.append(newLoss)
            
            # Generate accuracy stats on test data
            acc = accuracy(pred, yGold)
            accuracy_values.append(acc)
    
            # Re-assign values for variables
            diff = abs(newLoss - loss)
            loss = newLoss

            #generate print statements
            print("step %d, training accuracy %g, loss %g, change in loss %g"%(i, acc, newLoss, diff))


# How well do we perform on held-out test data?
print("final accuracy on test set: %s" %str(acc))