import torch
import torch.nn as nn
from Feed_Forward import FeedForward

d_model=512
layer_count=8

FF=FeedForward(d_model,layer_count)

input=torch.rand(32,1024,d_model)

output=FF.calculate(input)

print(output.shape)
print(output)
