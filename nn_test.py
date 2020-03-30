import tensorflow.keras.backend as k
import tensorflow as tf
import numpy as np

# Create Dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

Y = np.array([[0],
              [1],
              [1],
              [0]])

# Create to tensor
X = tf.convert_to_tensor(X, dtype=tf.float16)
Y = tf.convert_to_tensor(Y, dtype=tf.float16)

# Randomly initialise two weight matrics
# w1 is a matric with dims [2, 3]
# w2 is a matric with dims [3, 1]
# Coastin them to variable since value can be changed during optimisation
w1 = tf.Variable(np.random.randn(2, 3), dtype=tf.float16)
w2 = tf.Variable(np.random.randn(3, 1), dtype=tf.float16)

# Create a function that propagates X throught the network
@tf.function
def forward(X, w1, w2):
    X = tf.sigmoid(tf.matmul(X, w1))
    X = tf.sigmoid(tf.matmul(X, w2))
    return X


# Print
print(forward(X, w1, w2))
