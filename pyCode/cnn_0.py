import torch
from torch import nn

a = torch.tensor([1,2,3])

b = a.sum()
c = torch.stack((a,a))

d = a.float()

print(a,b,c,d)
print(id(a))
print(id(d))


print((1,3)+(2,4))


