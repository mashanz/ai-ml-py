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

import lamaproses
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

# From the previous sliide we can see that
# the model is just guessing. we need to implement
# a loss function that tells the optimizer how far off
# the prediction are from the actual target values.


@tf.function
def loss(predicted_y, target_y):
    return tf.reduce_mean(k.binary_crossentropy(target_y, predicted_y))

# Train Function's role to converge the model
# towards the actual target value


@tf.function
def train(inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        # Calculate loss
        current_loss = loss(forward(X, w1, w2), outputs)
    # find gradien between the loss and each weight matrix
    dW1, dW2 = tape.gradient(current_loss, [w1, w2])
    # backpropate and tune weights
    w1.assign_sub(learning_rate * dW1)
    w2.assign_sub(learning_rate * dW2)
    del tape

# Train the model for a number of iteration (epochs)
# Overtime the loss outputted onto the console will
# decrease, this means that your model is successfully
# training.


epochs = range(100000)
learning_rate = 0.1

for epochs in epochs:
    # Calculate loss to output.
    current_loss = loss(forward(X, w1, w2), Y)
    # Forward the input data and tweak weights.
    train(X, Y, learning_rate)
    print("Current Loss: {:2.5f}".format(current_loss))

# Test the model with a suitable domain
x_ = tf.convert_to_tensor(np.array([[0, 0]]), dtype=tf.float16)
print("0 XOR 0 -> {} ".format(forward(x_, w1, w2).numpy()))

x_ = tf.convert_to_tensor(np.array([[0, 1]]), dtype=tf.float16)
print("0 XOR 1 -> {} ".format(forward(x_, w1, w2).numpy()))

x_ = tf.convert_to_tensor(np.array([[1, 0]]), dtype=tf.float16)
print("1 XOR 0 -> {} ".format(forward(x_, w1, w2).numpy()))

x_ = tf.convert_to_tensor(np.array([[1, 1]]), dtype=tf.float16)
print("1 XOR 1 -> {} ".format(forward(x_, w1, w2).numpy()))
