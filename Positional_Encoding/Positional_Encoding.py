import sys
import torch
import math

# Configure stdout to use UTF-8 by default to correctly print Unicode on Windows.
sys.stdout.reconfigure(encoding='utf-8')

max_seq_len = 1024
d_model=512

PE = torch.zeros(max_seq_len,d_model)

for pos in range(max_seq_len):
    for i in range(d_model):
        if i%2==0:
            PE[pos,i] = math.sin(pos/(10000**(2/d_model)))
        else:
            PE[pos,i] = math.cos(pos/(10000**(2/d_model)))

save_path = "Positional_Encoding.pt"
torch.save(PE, save_path)
