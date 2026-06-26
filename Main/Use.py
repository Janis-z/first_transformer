import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.stdout.reconfigure(encoding='utf-8')

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

use_Transformer=Transformer(1)

use_Transformer.load_state_dict(torch.load(r"C:\Users\Janis\Downloads\Transformer\trained_transformer.pth"))


input=[["hello how"]]

print(use_Transformer.use(input))
