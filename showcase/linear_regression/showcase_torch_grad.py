import torch

x = torch.arange(4.0)
print(x)
x.requires_grad_(True)
y = 2 * torch.dot(x, x)
print(torch.dot(x, x))
print(y)

y.backward()
gd = x.grad

print(gd)
print(y)
# x.requires_grad_(True)
# y = 2 * torch.dot(x, x)
#y.backward()
#x.grad

# x = torch.arange(4.0)
# x.requires_grad_(True)
# y = 2 * torch.dot(x, x)
# y.backward()
# x.grad