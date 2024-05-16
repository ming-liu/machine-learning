import torch
import random


#
def create_data(weights, bias, sigma, num_examples):
    X = torch.normal(0, 1, (num_examples, len(weights)))
    y = torch.matmul(X, weights) + bias
    y += torch.normal(0, sigma, y.shape)
    print(y)
    print(y.reshape((-1, 1)))
    return X, y.reshape((-1, 1))


w = torch.tensor([5, 3.8])
b = 8.3
# features, labels =\
create_data(w, b, sigma=0.1, num_examples=1000)
