import random

import torch
import util


# reshape 成同样的结构很重要
def cost_function(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def linear_regression(features, weights, bias):
    return torch.matmul(features, weights) + bias


def data_iter(features, labels, batch_num):
    num_examples = len(features)
    index_list = list(range(num_examples))
    random.shuffle(index_list)

    for i in range(0, num_examples, batch_num):
        selected_list = index_list[i:min(i + batch_num, num_examples)]
        yield features[selected_list], labels[selected_list]


weights_true = torch.tensor([2.1, 1.7])
bias_true = torch.tensor(7.3)

features, labels = util.create_data(weights_true, bias_true, 0.1, 1000)
lr = 0.03
epoch_num = 20

weights = torch.tensor([0.1, 0.1], requires_grad=True)
bias = torch.tensor(0.1, requires_grad=True)

for epoch in range(epoch_num):
    for X, y in data_iter(features, labels, 100):
        batch_size = len(X)
        y_hat = linear_regression(X, weights, bias)
        loss = cost_function(y_hat, y)
        loss.sum().backward()

        with torch.no_grad():
            weights -= lr * weights.grad / batch_size
            bias -= lr * bias.grad / batch_size
            weights.grad.zero_()
            bias.grad.zero_()

    with torch.no_grad():
        loss = cost_function(linear_regression(features, weights, bias), labels).mean()
        print(weights, float(bias), float(loss))
