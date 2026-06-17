import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Decoder.Decoder import Decoder
from Encoder.Encoder import Encoder
from Linear_Layer.Linear_Layer import LinearLayer
from Input.Tokenizer import Tokenizer

from Training.Backpropagation import Backpropagation


from Input.Tokenizer import tokenizer


import torch
import torch.nn as nn




class Transformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.d_model=512
        self.context_size=1024
        self.num_heads=8
        self.num_layers=6
        self.batch_size=3
        self.vokab_size=50260

        self.Encoder1=Encoder(self.d_model,self.context_size,self.num_heads, self.num_layers,self.batch_size)
        self.Encoder2=Encoder(self.d_model,self.context_size,self.num_heads, self.num_layers,self.batch_size)
        self.Encoder3=Encoder(self.d_model,self.context_size,self.num_heads, self.num_layers,self.batch_size)

        self.Decoder1=Decoder(self.d_model,self.context_size,self.num_heads, self.num_layers,self.batch_size)
        self.Decoder2=Decoder(self.d_model,self.context_size,self.num_heads, self.num_layers,self.batch_size)
        self.Decoder3=Decoder(self.d_model,self.context_size,self.num_heads, self.num_layers,self.batch_size)

        self.Encoder_List=[self.Encoder1,self.Encoder2,self.Encoder3]
        self.Decoder_List=[self.Decoder1,self.Decoder2,self.Decoder3]

        self.LinearLayer1=LinearLayer(self.d_model,self.vokab_size)




    def train(self, input,Learningrate):

        Decoder_Embeddings,Decoder_Ids,Encoder_Embeddings,Encoder_Ids = Tokenizer.input_to_Embeddings(input,self.context_size)

        Positional_Encoding=torch.load("Positional_Encoding.pt")

        Encoder_input=Encoder_Embeddings+Positional_Encoding.unsqueeze(0)

        

        for Encoder in self.Encoder_List:
            Encoder_input=Encoder.forward(Encoder_input)
            

        Encoder_output=Encoder_input
        Decoder_input=Encoder_output

        

        for Decoder in self.Decoder_List:
            Decoder_input=Decoder.forward(Decoder_input,Encoder_output)

        

        output=self.LinearLayer1.forward(Decoder_input)

        output_probabilities = torch.softmax(output,dim=-1)

        

        #Encoder_Ids are the target
        Backpropagation().calculate(self.Decoder_List, self.Encoder_List, self.LinearLayer1, output_probabilities, Encoder_Ids,Encoder_Ids,Decoder_Ids,Learningrate,self.context_size, self.batch_size)








