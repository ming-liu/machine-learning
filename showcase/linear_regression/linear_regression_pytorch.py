import torch
import random
import matplotlib.pyplot as plt


# 模拟数据，增加噪音
def create_data(weights, bias, sigma, num_examples):
    X = torch.normal(0, 1, (num_examples, len(weights)))
    y = torch.matmul(X, weights) + bias
    y += torch.normal(0, sigma, y.shape)
    # (-1,1) -> from [1,2,3....] to [[1],[2],[3]...]
    return X, y.reshape((-1, 1))


def linear_reg(X, weights, bias):
    return torch.matmul(X, weights) + bias


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_examples)]
        yield features[batch_indices], labels[batch_indices]


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 1. create data
w = torch.tensor([5, 3.8])
b = 8.3
features, labels = create_data(w, b, sigma=0.1, num_examples=1000)
print('features.shape=', features.shape, ';labels.shape=', labels.shape)

plt.figure('Linear Regression')
plt.scatter(features[:, 0], labels, linewidth=2.0, color='blue', alpha=0.5)
plt.title('Linear Regression:')
plt.ylabel('label')
plt.xlabel('feature 0')
plt.show()

# 2. init
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# learning rate
lr = 0.03
loss = squared_loss
net = linear_reg
num_epochs = 5
batch_size = 100

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size=batch_size, features=features, labels=labels):
        # X和y的小批量损失
        l = loss(net(X, w, b), y)
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数

    with torch.no_grad():
        training_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(training_l.mean()):f}')
