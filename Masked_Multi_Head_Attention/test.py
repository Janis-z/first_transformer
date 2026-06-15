import torch
from Masked_Multi_Head_Attention import MaskedMultiHeadAttention

d_model=512
num_heads=8
batch_size=32

MMHA=MaskedMultiHeadAttention(batch_size,d_model,num_heads)

input=torch.rand(batch_size,1012,d_model)

output=MMHA.calculate(input,input,input)

print(output.shape)
print(output)


