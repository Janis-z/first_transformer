from sympy.parsing.sympy_parser import null
from rich.console import ThemeContext
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'additional_special_tokens': ['[SOS]', '[EOS]','[PAD]']})

class Tokenizer():

    @staticmethod
    def input_to_Embedings(input,context_size):
        
        #add start of sentence token
        tokens=["[SOS]"]
        tokens.extend(tokenizer.tokenize(input))
        
        #if the input is to long return
        if len(tokens)-context_size>0:
            return null

        #add EOS
        tokens.append("[EOS]")
        
        #add padding tokens to get to context size
        padding_needed=context_size-len(tokens)

        tokens.extend(["[PAD]"]*padding_needed)
        
        ids=tokenizer.convert_tokens_to_ids(tokens)
        
        #all embedings
        embedding = torch.load("Token_Embeddings.pt")

        #only return the embedings for the tokens in the input
        return embedding[ids],ids

    
    def input_to_targets(input,context_size,batch_size):

        tokens=tokenizer.tokenize(input)
        
        #if the input is to long return
        if len(tokens)-context_size>0:
            return null

        #add padding tokens to get to context size
        padding_needed=context_size-len(tokens)

        tokens.extend(["[PAD]"]*padding_needed)

        ids=tokenizer.convert_tokens_to_ids(tokens)

        ids_tensor = torch.tensor(ids)

        return ids_tensor



        
        

