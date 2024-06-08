import torch


def create_data(weights, bias, sigma, num_examples):
    """
    模拟数据，增加噪音
    :param weights:
    :param bias:
    :param sigma:
    :param num_examples:
    :return:
    """
    X = torch.normal(0, 1, (num_examples, len(weights)))
    y = torch.matmul(X, torch.reshape(weights, (-1, 1))) + bias
    y += torch.normal(0, sigma, y.shape)
    return X, y
