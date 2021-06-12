import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress warning that it can't use CUDA
import tensorflow as tf

Matrix_one = tf.constant([[2,3],[3,4]])
Matrix_two = tf.constant([[2,3],[3,4]])

@tf.function
def mathmul():
  return tf.matmul(Matrix_one, Matrix_two)


mul_operation = mathmul()

print ("Defined using tensorflow function :")
print(mul_operation)