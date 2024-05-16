import torch

print('###### part1, tensor base ######')
print('torch.tensor(1):', torch.tensor(1))
print('torch.tensor(2.0):', torch.tensor(2.0))
print('torch.tensor([1, 2, 3]):', torch.tensor([1, 2, 3]))
print('torch.tensor([[1, 2, 3], [4, 5, 6]]):', torch.tensor([[1, 2, 3], [4, 5, 6]]))

print('###### part2, scalar * scalar ######')
x1 = torch.tensor(3.0, requires_grad=True)
x2 = torch.tensor(5.0, requires_grad=True)
y = (x1 + 1) * (x2 + 3)
y.backward()
print(x1.grad)  # ∂y/∂x1 = x2 + 3 = 8
print(x2.grad)  # ∂y/∂x2 = x1 + 1 = 4

print('###### part3, scalar * scalar * scalar ######')
x1 = torch.tensor(3.0, requires_grad=True)
x2 = torch.tensor(5.0, requires_grad=True)
y = (x1 + 1) * (x1 + 2) * (x2 + 3)
y.backward()
print(x1.grad)  # ∂y/∂x1 = (x2 + 3) * (2x1 + 3) = 8 * 9 = 72
print(x2.grad)  # ∂y/∂x2 = (x1 + 1) * (x1 + 2) = 20

print('###### part4, [1, 2, 3] * [1, 1, 1].transpose ######')
x1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
x2 = torch.tensor([[1.0], [1.0], [1.0]], requires_grad=True)
y = torch.matmul(x1, x2)
y.backward()
# 标量对向量求导，结果是向量，形状应该是X的转置。y分别对每个x求导。
# y = x1_1 * x2_1 + x1_2 * x2_2 + x1_3 * x3_3
# ∂y/∂x1_1 = x2_1
# ∂y/∂x1_2 = x2_2
# ∂y/∂x1_3 = x2_3
# [x2_1,x2_2,x2_3] 的转置 ,pytorch并没有很严谨转置。
# pytorch的实现是：标量对向量求导，形状为向量的形状。（没有转置）
print(x1.grad)
# 标量对向量求导，结果是向量，形状应该是X的转置。y分别对每个x求导。
# [1, 2, 3]
print(x2.grad)

print('###### part5, net ######')
x11 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
x12 = torch.tensor([[1.0], [1.0], [1.0]], requires_grad=True)
x21 = x11.matmul(x12)
x21.requires_grad_(True)

x13 = torch.tensor([[1.0, 2], [3, 4]], dtype=float, requires_grad=True)
x14 = torch.tensor([[1.0, 2], [3, 4]], dtype=float, requires_grad=True)
x14.requires_grad_(True)
x22 = x13.matmul(x14)
x22.requires_grad_(True)

x3 = x22 * x21
x3.requires_grad_(True)

# x21 = 6; x22 = [7 10;15 22] ; x3 = [42 60;90 132]; y = 324
y = x3.sum()
print('x21:', x21)
print('x22:', x22)
print('x3:', x3)
print('y:', y)

x3.retain_grad()
x21.retain_grad()
x22.retain_grad()
y.backward()

# x3 = [42 60;90 132] y = sum(x3)
# y = x3_1 + x3_2 + x3_3 + x3_4; ∂y/∂x3 = [1 1;1 1]
print('x3.grad: ', x3.grad)
# y = f(x3_1) + f(x3_2) + f(x3_3) + f(x3_4);
# x21 = 6; x22 = [7 10;15 22] ;
# x3_1 = x21 * x22_1
# x3_2 = x21 * x22_2
# x3_3 = x21 * x22_3
# x3_4 = x21 * x22_4
# x21,y are both scalar
# ∂y/∂x21 = ∂f(x3_1)/∂x21 + ∂f(x3_2)/∂x21 + ∂f(x3_3)/∂x21 + ∂f(x3_4)/∂x21
#         = ∂f(x3_1)/∂x3_1 * ∂x3_1/∂x21 + ...
#         = ∂y/∂x3_1 * ∂x3_1/∂x21 + ∂y/∂x3_2 * ∂x3_2/∂x21 + ∂y/∂x3_3 * ∂x3_3/∂x21 + ...
#         = 1 * x22_1 + 1 * x22_2 + 1 * x22_3 + 1 * x22_4
#         这里 ∂y/∂x3_1 = ∂f(x3_1)/∂x3_1 ,其他几项对x3_1求导都是0
print('x21.grad', x21.grad)

# x21 = 6; x22 = [7 10;15 22] ;
# x3_1 = x21 * x22_1
# x3_2 = x21 * x22_2
# x3_3 = x21 * x22_3
# x3_4 = x21 * x22_4
# ∂y/∂x22_1 = ∂y/∂x3_1 * ∂x3_1/∂x22_1 = 6
# 同理，其他几项
print('x22.grad', x22.grad)

