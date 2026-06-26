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
from torch.utils.data import DataLoader, IterableDataset

from Transformer import Transformer


train_Transformer=Transformer(2)

#load old transformer values
#train_Transformer.load_state_dict(torch.load(r"C:\Users\Janis\Downloads\Transformer\trained_transformer.pth"))

#train on only a bit because my laptop dies 
for i in range(50):
    input=[["hello how are u?"]]
    input.append("hello how are u?")
    #0.000005 because above is to much and values turn to nan
    train_Transformer.train(input,0.000005)   
    torch.save(train_Transformer.state_dict(), r"C:\Users\Janis\Downloads\Transformer\trained_transformer.pth")
    print(i)



#wikipedia trainer under here

# from datasets import load_dataset
# 
# # Load Wikipedia en
# dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
# 
# 
# #chunksize 600 to get around 1024 batches
# chunk=600
# input=[]
# for num,article in enumerate(dataset["train"]):
# 
#     article_length=len(article["text"])
# 
#     for text_start in range(0,article_length, chunk):
# 
#         input.append(article["text"][text_start:text_start+chunk])
# 
#         if len(input)==3:
#             train_Transformer.train(input,0.04)
#             input=[]
#         print(text_start)
# 
#     print(num)
#     torch.save(train_Transformer.state_dict(), r"C:\Users\Janis\Downloads\Transformer\trained_transformer.pth")
 
