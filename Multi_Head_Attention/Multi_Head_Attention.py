import sys
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    
    def __init__(self, batch_size, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.batch_size = batch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        

        #disable biases to keep simple 
        self.Wq= nn.Linear(d_model,d_model, bias=False)
        self.Wk= nn.Linear(d_model,d_model, bias=False)
        self.Wv= nn.Linear(d_model,d_model, bias=False)
        self.Wo= nn.Linear(d_model,d_model, bias=False)

    def calculate(self,Q,K,V):

        self.cache = {
            'Q': Q,
            'K': K,
            'V': V
        }

        #first * the weights with dot product
        Qw = self.Wq(Q)
        Kw = self.Wk(K)
        Vw = self.Wv(V)

        self.cache.update = {
            'Qw': Qw,
            'Kw': Kw,
            'Vw': Vw
        }


        seq_len = Qw.shape[1]

        #split into multiple heads
        #first put seq_len first then switch with num_heads to get (num_heads, seq_len, head_size)
        #permute says what position the values go by there previouse order
        Qw_split = Qw.view(self.batch_size, seq_len, self.num_heads, self.head_size).permute(0,2,1,3)
        Kw_split = Kw.view(self.batch_size, seq_len, self.num_heads, self.head_size).permute(0,2,1,3)
        Vw_split = Vw.view(self.batch_size, seq_len, self.num_heads, self.head_size).permute(0,2,1,3)
        

        self.cache.update({
            'Qw_split': Qw_split,
            'Kw_split': Kw_split,
            'Vw_split': Vw_split
        })

        #Calculate Attention
        #Transpose to make (seq,head) to (head,seq) to get (batch,heads,seq,seq)
        Q_K = Qw_split @ Kw_split.transpose(-2,-1)

        #dim=-1 = softmax on the last dimension, the keys
        Q_K_Soft=torch.softmax(Q_K / math.sqrt(self.head_size),dim=-1)

        #dot with V to get the context but still split
        H_Split=Q_K_Soft @ Vw_split

        #allign the values back
        #permute alligns, contiguous aligns in storage(permute doesnt do that), view just puts all dimensions together like stacking all the single heads, to get d_model back
        H=H_Split.permute(0,2,1,3).contiguous().view(self.batch_size, seq_len, self.d_model)

        #calculate output
        output=self.Wo(H)

        self.cache.update({
            'Q_K':Q_K,
            'Q_K_Soft':Q_K_Soft,
            'H_Split':H_Split,
            'H':H,
            'output':output
        })

        return output
