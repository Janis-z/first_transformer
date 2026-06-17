from sympy.parsing.sympy_parser import null
from rich.console import ThemeContext
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'additional_special_tokens': ['[SOS]', '[EOS]','[PAD]']})

class Tokenizer():

    @staticmethod
    def input_to_Embeddings(input,context_size):
        #all embedings
        embedding = torch.load("Token_Embeddings.pt")

        decoder_Embedings=[]
        encoder_Embedings=[]
        decoder_Ids=[]
        encoder_Ids=[]

        for Batch_Count,Batch in enumerate(input):
        
            #add start of sentence token
            tokens=["[SOS]"]
            tokens.extend(tokenizer.tokenize(Batch))
            
            #if the input is to long return
            if len(tokens)-context_size>0:
                return None

            #Encoder doesnt need SOS so delete it
            encoder_tokens=tokens.copy()

            del encoder_tokens[0]

            #add EOS only for encoder 
            encoder_tokens.append("[EOS]")
            
            #add padding tokens to get to context size
            padding_needed=context_size-len(tokens)

            padding_needed_encoder=context_size-len(encoder_tokens)

            tokens.extend(["[PAD]"]*padding_needed)

            encoder_tokens.extend(["[PAD]"]*padding_needed_encoder)
            
            decoder_Ids.append(tokenizer.convert_tokens_to_ids(tokens))
            
            encoder_Ids.append(tokenizer.convert_tokens_to_ids(encoder_tokens))
 

       
        #make ids to tensor
        decoder_Ids=torch.tensor(decoder_Ids)
        encoder_Ids=torch.tensor(encoder_Ids)


        return (torch.tensor(embedding[decoder_Ids]),decoder_Ids,
                torch.tensor(embedding[encoder_Ids]),encoder_Ids)

    
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



        
        

