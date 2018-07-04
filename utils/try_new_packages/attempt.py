import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()


a = [1, 2, 1, 1, 1]
b = tf.one_hot(a, depth=4)
print(b)