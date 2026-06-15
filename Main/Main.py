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
Decoder1=Decoder(d_model,context_size,num_heads, num_layers,batch_size)
LinearLayer1=LinearLayer(d_model,vokab_size)

input_text = input("Enter your prompt: ")

Embedings,Ids = Tokenizer.input_to_Embedings(input_text,context_size)

Positional_Encoding=torch.load("Positional_Encoding.pt")

input=Embedings+Positional_Encoding.unsqueeze(0)

print(input.shape)

Encoder_output=Encoder1.forward(input)

Decoder_output=Decoder1.forward(Encoder_output,Encoder_output)

output=LinearLayer1.forward(Decoder_output)

print(output.shape)

output_probabilities = torch.softmax(output,dim=-1)

best_token_indices = torch.argmax(output_probabilities, dim=-1)

token_ids = best_token_indices[0].tolist()

decoded_text = tokenizer.decode(token_ids)
print(decoded_text)
