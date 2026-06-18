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


train_Transformer=Transformer()

#load old transformer values
#train_Transformer.load_state_dict(torch.load(r"C:\Users\Janis\Downloads\Transformer\trained_transformer.pth"))
# Falls nicht schon oben importiert, hier die Klasse definieren:
class MyIterableDataset(IterableDataset):
    def __init__(self, generator_function):
        self.generator_function = generator_function
        
    def __iter__(self):
        return self.generator_function()


#add wikipedia
def get_wikipedia_batches(batch_size, context_size, lang="en"):
    print(f"Lade Wikipedia ({lang}) im Streaming-Modus...")
    # Lädt die Daten live aus dem Netz, ohne die Festplatte zu füllen
    dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)

    def generator():
        buffer = []
        for row in dataset:
            text = row["text"]

            woerter = text.split()

            buffer.extend(woerter)
            
            # 2. Sobald der Buffer groß genug für eine Context Size ist, Stücke abschneiden
            while len(buffer) >= context_size:
                chunk = buffer[:context_size]
                buffer = buffer[context_size:]
                yield torch.tensor(chunk)
    
    iterable_dataset = MyIterableDataset(generator)
    # DataLoader baut automatisch die Batches im Hintergrund zusammen
    dataloader = DataLoader(iterable_dataset, batch_size=batch_size)
    return dataloader



batch_loader = get_wikipedia_batches(batch_size=train_Transformer.batch_size,context_size=train_Transformer.context_size-2)


for step, batch in enumerate(batch_loader):
    print(batch)

print(input)
train_Transformer.train(input,1)

print("lool")
#dataset= load_dataset("wikipedia", "20220301.en", split="train",streaming=True)

torch.save(train_Transformer.state_dict(), r"C:\Users\Praktikant\Downloads\first_transformer\trained_transformer.pth")