from typing import List
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from compress import compress
from decompress import decompress
from arithmetic_coding import read_and_decode, encode_and_store

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to("cuda") # type: ignore
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

M = 16

print("Reading data.txt...")
#  read the data.txt file
with open("alice_in_wonderland.txt", "r") as file:
    text = file.read()

# Tokenize the input text
input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda") # type: ignore

print("Compressing...")
# Compress the input text
ranks = compress(input_ids, model, M)

print("Applying Arithmetic Coding...")
# # Apply Arithmetic Coding to the ranks
encode_and_store(ranks, "compressed.bin")

print("Reading compressed.bin and decoding...")
# # Read and decode the compressed file
decoded_ranks = read_and_decode("compressed.bin")

print("Decompressing...")
# # Decompress the ranks
decompressed_text = decompress(ranks, input_ids[0][:M], tokenizer, model, M)

print("Saving decompressed text to 'decompressed.txt'...")
# Save the decompressed text to a file
with open("decompressed.txt", "w") as file:
    file.write(decompressed_text)

print("Decompressed text saved to 'decompressed.txt'")