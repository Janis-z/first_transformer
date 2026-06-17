from Masked_Multi_Head_Attention.Masked_Multi_Head_Attention import MaskedMultiHeadAttention
from Multi_Head_Attention.Multi_Head_Attention import MultiHeadAttention
from Norm.Norm import Norm
from Feed_Forward.Feed_Forward import FeedForward
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, d_model,token_count,num_heads, num_layers,batch_size):
        super().__init__()
        
        self.mmha = MaskedMultiHeadAttention(batch_size, d_model, num_heads)
        self.mha = MultiHeadAttention(batch_size, d_model, num_heads)
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)
        self.feedforward = FeedForward(d_model, num_layers)

        self.cache={
            "add_output_1":None,
            "add_output_2":None,
            "add_output_3":None,
            "mmha_output":None,
            "mha_output":None,
            "feed_forward_output":None,
        }

    def forward(self,input,encoder_output):

        print("hallos")
        
        norm_input_1=self.norm1.apply_norm(input)
        mmha_output=self.mmha.calculate(norm_input_1,norm_input_1,norm_input_1) 

        add_output_1=mmha_output+input

        self.cache["add_output_1"]=add_output_1
        self.cache["mmha_output"]=mmha_output

        print("waaasss")
        
        #Q from Decoder rest from Encoder
        norm_input_2=self.norm2.apply_norm(add_output_1)
        mha_output=self.mha.calculate(norm_input_2,encoder_output,encoder_output)

        add_output_2=mha_output+add_output_1

        self.cache["add_output_2"]=add_output_2
        self.cache["mha_output"]=mha_output

        norm_input_3=self.norm3.apply_norm(add_output_2)
        feed_forward_output=self.feedforward.calculate(norm_input_3)

        add_output_3=feed_forward_output+add_output_2

        self.cache["add_output_3"]=add_output_3
        self.cache["feed_forward_output"]=feed_forward_output

        return add_output_3



        

        

