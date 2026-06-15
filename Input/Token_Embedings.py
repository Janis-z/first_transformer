import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Configure stdout to use UTF-8 by default to correctly print BPE special characters (like Ġ) on Windows.
sys.stdout.reconfigure(encoding='utf-8')

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Define custom special tokens:
# [SOC] -> Start of Context (marks the beginning of our input/context sequence)
# [EOC] -> End of Context (marks the end of our input/context sequence)
special_tokens_dict = {'additional_special_tokens': ['[SOC]', '[EOC]','[PAD]']}

# Add the special tokens to the tokenizer vocabulary
num_added = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added} special tokens to the tokenizer.")

# Verify the token IDs of the newly added special tokens
soc_id = tokenizer.convert_tokens_to_ids('[SOS]')
eoc_id = tokenizer.convert_tokens_to_ids('[EOS]')
pad_id = tokenizer.convert_tokens_to_ids('[PAD]')
print(f"[SOS] token has ID: {soc_id}")
print(f"[EOS] token has ID: {eoc_id}\n")

# Define vocabulary size and embedding dimensions (512 per token)
# vocab_size now increases from 50,257 to 50,259 to accommodate the new special tokens.
vocab_size = len(tokenizer)
embedding_dim = 512

# Initialize the full token embedding matrix for the entire vocabulary
# This now contains 50,259 rows (one for each token, including [SOC] and [EOC]) and 512 columns.
token_embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# Extract the entire embedding weight matrix (shape: [vocab_size, embedding_dim])
full_embedding_matrix = token_embedding_layer.weight

# Save the entire vocabulary embedding matrix to disk
save_path = "Token_Embeddings.pt"
torch.save(full_embedding_matrix, save_path)

print(f"Total vocabulary size (including custom special tokens): {vocab_size}")
print(f"Generated entire embedding matrix shape: {full_embedding_matrix.shape}")
print(f"Saved complete embedding matrix to: {save_path}")

