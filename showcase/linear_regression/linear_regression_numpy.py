import numpy as np
from mxnet import  npx
import matplotlib.pyplot as plt


# 简介版
def create_data(weights, bias, sigma, num_examples):
    X = np.random.normal(0, 1, (num_examples, weights.size))
    y = X.dot(weights) + bias
    y += np.random.normal(0, sigma, y.shape)
    return X, y.reshape((-1, 1))


def linear_reg(X, weights, bias):
    return X.dot(weights) + bias


def squared_loss(y_hat, y):
    return (y_hat - y.reshap(y_hat.shape)) ** 2 / 2


def data_iter(batch_size, fetures, labels):
    num_examples = len(fetures)
    indices = list(range(num_examples))
    np.random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i : min(i + batch_size, num_examples)]
        yield fetures[batch_indices], labels[batch_indices]

# 1. create data
w = np.array([5, 3.8])
b = 8.3
features, label = create_data(w, b, sigma=0.1, num_examples=100)


plt.figure('Linear Regression')
plt.scatter(features[:, 0], label, linewidth=2.0, color='blue', alpha=0.5)
plt.title('Linear Regression:')
plt.ylabel('label')
plt.xlabel('feature 0')
plt.show()

# 2. init
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()

# learning rate
alpha = 0.03
loss = squared_loss
net = linear_reg
num_epochs = 3

# print(features)
for X, y in data_iter(10, features, labels=label):
    print(X, '\n', y)
    break

# range(0, num_examples, batch_size)
# print(range(0, 100, 10))
#for i in range(0, 100, 10):
#    print(i)


