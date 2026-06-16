from transformers import AutoTokenizer
from Tokenizer import Tokenizer


input=[]
input.append("test")

input.append("hi")

decoder_embedings,decoder_ids,encoder_embedings,encoder_ids=Tokenizer.input_to_Embedings(input,1024)

print(decoder_embedings.shape,decoder_ids.shape)