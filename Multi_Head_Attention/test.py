import torch
from Multi_Head_Attention import MultiHeadAttention

d_model=512
num_heads=8
batch_size=32

MHA=MultiHeadAttention(batch_size,d_model,num_heads)

input=torch.rand(batch_size,1012,d_model)

output=MHA.calculate(input,input,input)

print(output.shape)
print(output)

