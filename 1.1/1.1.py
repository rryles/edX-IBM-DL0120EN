import tensorflow as tf

print (tf.__version__)

a = tf.constant([2], name = 'constant_a')
b = tf.constant([3], name = 'constant_b')

@tf.function
def add(a, b):
    c = tf.add(a, b)
    print(c)
    return c

result = add(a, b)
tf.print(result[0])

