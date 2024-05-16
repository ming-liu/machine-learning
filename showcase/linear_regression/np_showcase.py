import numpy as np
import matplotlib.pyplot as plt


# 随机序列、整体分布随机序列
def showRandomNormalPlot():
    # 随机生成一个10以内的整数
    a = np.random.randint(10)
    print('10以内的随机数:', a)
    print(type(a))

    # 产生均匀分布的样本值(0-1,100个)
    a = np.random.rand(100)
    print('均匀分布的100个数字:', a)
    print(type(a))
    plt.figure('Random numbers')
    plt.hist(a, bins=20, color='blue', alpha = 0.5)
    plt.title('Random numbers:')
    plt.ylabel('Frequence')
    plt.xlabel('Values')
    plt.show()

    # 生成一个长度为100的符合正态分布的随机数列
    a = np.random.randn(100)
    print(type(a))
    print('正态分布的100个数字:', a)
    plt.figure('Normal Distribution')
    plt.hist(a, bins=20, color='blue', alpha = 0.5)

    plt.title('Normal distribution numbers:')
    plt.ylabel('Frequence')
    plt.xlabel('Values')
    plt.show()

    # 生成一个10*10的符合随机分布的随机数列，符合正态分布(Normal distribution)
    a = np.random.randn(10, 10)
    print(type(a))
    print('正态分布的10*10个数字:', a)

    # 均值和标准差
    # mean and standard deviation
    mean,sdand_dev = 0,0.1
    # a = np.random.normal(loc=mean, scale=stddev, size=1)
    a = np.random.normal(mean, sdand_dev, 1000)
    print('mean,standard deviation,size = ' ,mean ,sdand_dev, a.size)
    plt.figure('Normal Distribution 2')
    plt.hist(a, bins=20, color='blue', alpha = 0.5, density=True)

    # count, bins, ignored = plt.hist(s, 30, density=True)
    # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    #                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
    #          linewidth=2, color='r')

    plt.title('Normal distribution numbers(mean=%s,standard deviation=%s):' % (mean, sdand_dev))
    plt.ylabel('Frequency')
    plt.xlabel('Values')
    plt.show()
    # d2l.set_figsize()
    # d2l.plt.scatter(features[:, (1)].asnumpy(), labels.asnumpy(), 1);


def showMetrixOperations():
    x = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
    bias = [13,14,15,16]
    x1 = np.column_stack((x, bias))
    x2 = np.insert(x,obj=0 ,values=bias, axis=1)
    x3 = np.insert(x,obj=1 ,values=bias, axis=1)
    print(x1)
    print(x2)
    print(x3)

    print(np.ones(5, dtype=int))
    print(np.ones((5,3), dtype=int))

    x = np.random.normal(0, 1, (5 , 2))
    print(x)
    print('###第二列')
    print(x[:,1])
    print('###+3')
    print(x[:,1] + 3)
    x = x[:,1] + 3
    x = x.reshape((-1,1))
    print('reshape(-1,1)')
    print(x)
    print(x.shape)
    

def create_data(weights, bias, std_devi, sizeOfData):
    print(weights.size)
    # data_X = np.array(sizeOfData, weights.size)
    # 无论行和列都是正态分布,无论行列
    # x = np.random.normal(0, 1, (weights.size , sizeOfData)).transpose()
    x = np.random.normal(0, 1, (sizeOfData , weights.size))
    # bias = np.random.normal(0, 0.1, sizeOfData)
    # x = np.column_stack((x, np.ones(sizeOfData)))
    print(x.shape)
    print(np.ones(sizeOfData))
    x = np.column_stack((x, np.ones(sizeOfData, dtype=int)))
    print(x.shape)
    print(x)
    weights = np.append(weights, bias)
    print(np.append(weights, bias))

    print(x.shape)
    print()
    y = x.dot(weights)
    print(y)
    return 123


showMetrixOperations()
# showRandomNormalPlot()
w = np.array([5, 3.8])
b = 8.3
std_devi = 0.1
# create_data(w, b, std_devi, 100)
print('[1,2,3] - [1,1,1]')
y = np.array([1, 2, 3]) - np.array([1, 1, 1])
print(y)
print(y ** 2 / 2)