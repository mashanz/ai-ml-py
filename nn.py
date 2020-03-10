# Simple Neural Network TensorFlow

'''
We will be building a simple Multi-layer Perceptron with one single hidden layer

Architecture:

        One Input Layer     (2 features)
        One Hidden Layer    (3 features)
        One Output Layer    (1 feature)

we will tackle the XOR Gate Problem, one can say it is the 'Hello World!' of building a neural net before MNIST.

The objective is to create a model (neural net) that behaves asn an XOR Gate:
0 XOR 0 -> 0
0 XOR 1 -> 1
1 XOR 1 -> 0
'''

import TensorFlow as tf
import numpy as np

# Create Dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

Y = np.array([[0],
              [1],
              [1],
              [0],])

# Create to tensor
X = tf.convert_to_tensor(X, dtype=tf.float16)
Y = tf.convert_to_tensor(Y, dtype=tf.float16)

# Randomly initialise two weight matrics
# w1 is a matric with dims [2, 3]
# w2 is a matric with dims [3, 1]
# Coastin them to variable since value can be changed during optimisation
w1 = tf.Variable(np.random.randn(2, 3), dtype=tf.float16)
w2 = tf.Variable(np.random.randn(3, 1), dtype=tf.float16)

# Create a function that propagates