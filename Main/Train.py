import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Decoder.Decoder import Decoder
from Encoder.Encoder import Encoder
from Linear_Layer.Linear_Layer import LinearLayer
from Input.Tokenizer import Tokenizer

from Input.Tokenizer import tokenizer

import torch
import torch.nn as nn


d_model=512
context_size=1024
num_heads=8
num_layers=6
batch_size=1
vokab_size=50259

Encoder1=Encoder(d_model,context_size,num_heads, num_layers,batch_size)
Encoder2=Encoder(d_model,context_size,num_heads, num_layers,batch_size)
Encoder3=Encoder(d_model,context_size,num_heads, num_layers,batch_size)
Encoder4=Encoder(d_model,context_size,num_heads, num_layers,batch_size)
Encoder5=Encoder(d_model,context_size,num_heads, num_layers,batch_size)
Encoder6=Encoder(d_model,context_size,num_heads, num_layers,batch_size)

Decoder1=Decoder(d_model,context_size,num_heads, num_layers,batch_size)
Decoder2=Decoder(d_model,context_size,num_heads, num_layers,batch_size)
Decoder3=Decoder(d_model,context_size,num_heads, num_layers,batch_size)
Decoder4=Decoder(d_model,context_size,num_heads, num_layers,batch_size)
Decoder5=Decoder(d_model,context_size,num_heads, num_layers,batch_size)
Decoder6=Decoder(d_model,context_size,num_heads, num_layers,batch_size)

Encoder_List=[Encoder1,Encoder2,Encoder3,Encoder4,Encoder5,Encoder6]
Decoder_List=[Decoder1,Decoder2,Decoder3,Decoder4,Decoder5,Decoder6]

LinearLayer1=LinearLayer(d_model,vokab_size)






Decoder_Embedings,Ids = Tokenizer.input_to_Embedings(input_text,context_size)

Encoder_Embedings=Decoder_Embedings

Positional_Encoding=torch.load("Positional_Encoding.pt")

Decoder_input=Decoder_Embedings+Positional_Encoding.unsqueeze(0)

print(input.shape)

for Encoder in Encoder_List:
    input=Encoder.forward(input)

Encoder_output=input

for Decoder in Decoder_List:
    input=Decoder.forward(input,Encoder_output)



Decoder_output=Decoder1.forward(Encoder_output,Encoder_output)

output=LinearLayer1.forward(Decoder_output)

print(output.shape)

output_probabilities = torch.softmax(output,dim=-1)


