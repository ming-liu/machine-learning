import torch
import torch.nn as nn
import util

from torch.utils import data


def get_loader(features, labels, batch_size, is_training=True):
    dataset = data.TensorDataset(features, labels)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_training)


weights_true = torch.tensor([3.8, 6.7])
bias_true = torch.tensor(7)

# 4大项： 数据、模型、损失函数、优化策略
features, labels = util.create_data(weights_true, bias_true, 0.1, 1000)
loader = get_loader(features, labels, 100)

linear = nn.Linear(2, 1)
linear.weight.data.normal_(0, 0.1)
linear.bias.data.fill_(0)
net = nn.Sequential(linear)

loss = nn.MSELoss()
sgd = torch.optim.SGD(net.parameters(), lr=0.03)

epoch_nums = 10
for epoch in range(epoch_nums):
    for X, y in loader:
        loss(net(X), y).backward()
        sgd.step()
        sgd.zero_grad()
    cost = loss(net(features), labels)
    print(f'{epoch + 1} loss {cost:f}')
