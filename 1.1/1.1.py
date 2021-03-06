import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress warning that it can't use CUDA
import tensorflow as tf

v = tf.Variable(0)

@tf.function
def increment(v):
    v = tf.add(v, 1)
    return v

for _ in range(3):
    v = increment(v)
    print(v)

a = tf.constant([5])
b = tf.constant([2])
c = tf.add(a,b)
d = tf.subtract(a,b)


print ('c =: %s' % c)
    
print ('d =: %s' % d)