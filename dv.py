import matplotlib.pyplot as plt
import tensorflow as tf


def f(x):
    return x


x = [1, 2, 3, 4, 5, 6]
y = list(map(f, x))

plt.plot(x, y, 'r-')
plt.plot(x, y, 'b-')
plt.plot(x, y, 'g-')
plt.plot(x, y, 'rx-')
plt.show()

classes = ["Class A", "Class B", "Class C"]
values = [0.15, 0.8, 0.05]

plt.bar(classes, values)
plt.title("Softmax Output")
plt.xlabel("Classes")
plt.ylabel("Probability")
plt.show()

train_loss_results = []
train_accuracy_results = []

num_epoch = 200

for epoch in range(num_epoch):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

fig, axes = plt.subplot(2, sharex=True, figsize=(12, 8))
fig.subtitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()
