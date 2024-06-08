import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# from d2l import torch as d2l

trans = torchvision.transforms.ToTensor()
print(trans, type(trans))

mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)


def describe_data(mnist_train):
    print('FashionMNIST train dataset: .............')
    print('train, type:', type(mnist_train), ';len=', len(mnist_train))

    print('first row : ')
    first_row = mnist_train[0]
    print('type:', type(first_row), ';len=', len(first_row))
    print('first_row[0].shape:', first_row[0].shape)
    print('first_row[1]:', first_row[1])


def get_fashion_mnist_labels(labels):  # @save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


describe_data(mnist_train)

# axes : 2 * 2 numpy.ndarray<Axes>
figure, axes = plt.subplots(nrows=2, ncols=2)
# numpy.ndarray<Axes>
axes = axes.flatten()

index = 0
for axis in axes:
    X, y = mnist_train[index]
    axis.imshow(torch.reshape(X, (28, 28)).numpy())
    index += 1

# plt.imshow(torch.reshape(X, (28, 28)).numpy())
# X, y = mnist_train[1]
# plt.imshow(torch.reshape(X, (28, 28)).numpy())
# X, y = mnist_train[2]
# plt.imshow(torch.reshape(X, (28, 28)).numpy())
# X, y = mnist_train[3]
# plt.imshow(torch.reshape(X, (28, 28)).numpy())

plt.show()

X = torch.reshape(X, (28, 28))
i = zip(X)
print(X.shape)
