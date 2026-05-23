import torch
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(42)

# Load the text file
with open("lettre_dune_inconnue.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Build vocabulary: sorted list of unique characters
chars = sorted(list(set(text)))
caracters_nbr = len(chars)
print(len(text))        # Total number of characters in the text
print(caracters_nbr)    # Number of unique characters (vocabulary size)

# Create mappings: character -> index and index -> character
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# encode: convert a string to an integers
encode = lambda s: [stoi[c] for c in s]
# decode: convert an integers a string
decode = lambda l: ''.join([itos[i] for i in l])
# convert the entire text into a tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)

# Define context window size and number of sequences per batch
block_size = 8
batch_size = 4

def get_batch(data, batch_size, block_size):
    """
    Sample a random batch of input/target pairs from the dataset.
    Each input x is a sequence of block_size characters,
    and y is the same sequence shifted by one (next character prediction).
    """
    x = []
    y = []
    for i in range(batch_size):
        # Pick a random starting index
        start = torch.randint(0, len(data) - block_size - 1, (1,)).item()
        x.append(data[start: block_size + start])           # input sequence
        y.append(data[start + 1: block_size + start + 1])   # Target sequence (shift by 1)
    return torch.stack(x), torch.stack(y)

# generate a batch of data
x, y = get_batch(data, batch_size, block_size)

# Create an embedding layer: maps each character index to a 32 dimensional vector
embedding_layer = nn.Embedding(num_embeddings=len(chars), embedding_dim=32)

# pass the batch through the embedding layer
# Output shape: (batch_size, block_size, embedding_dim) -> (4, 8, 32)
vecteur = embedding_layer(x)
print(vecteur)