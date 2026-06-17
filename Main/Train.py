import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Decoder.Decoder import Decoder
from Encoder.Encoder import Encoder
from Linear_Layer.Linear_Layer import LinearLayer
from Input.Tokenizer import Tokenizer

from Input.Tokenizer import tokenizer

from datasets import load_dataset

import torch
import torch.nn as nn

from Transformer import Transformer


train_Transformer=Transformer()

#load old transformer values
train_Transformer.load_state_dict(torch.load(r"C:\Users\Janis\Downloads\Transformer\trained_transformer.pth"))


input=[("hi")]

for i in range(2):
    input.append("hi")

print(input)
train_Transformer.train(input,1)


#dataset= load_dataset("wikipedia", "20220301.en", split="train",streaming=True)

torch.save(train_Transformer.state_dict(), r"C:\Users\Janis\Downloads\Transformer\trained_transformer.pth")