import torch
import torch.nn as nn
from Linear_Layer import LinearLayer


d_model=512
vokab_size=10000

LL=LinearLayer(d_model,vokab_size)

input=torch.rand(32,1024,d_model)

output=LL.calculate(input)

print(output.shape)

