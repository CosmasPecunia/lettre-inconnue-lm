from math import sqrt
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
#print(len(text))        # Total number of characters in the text
#print(caracters_nbr)    # Number of unique characters (vocabulary size)

# Create mappings: character -> index and index -> character
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# encode: convert a string to an integers
encode = lambda s: [stoi[c] for c in s]
# decode: convert an integers a string
decode = lambda l: ''.join([itos[i] for i in l])
# convert the entire text into a tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)

split = int(0.9*len(data))
train_data = data[:split]
val_data = data[split:]

# Define context window size and number of sequences per batch
block_size = 32
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

class Model_language(nn.Module):
    def __init__(self, vocab_size, block_size, embedding_dim):
        super().__init__()

        # each character becomes a vector of size embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # same but for the position, so the model knows where the character is
        # without this it would just see characters with no order
        self.position = nn.Embedding(block_size, embedding_dim)

        # the 3 layers for attention
        # Q : what i'm looking for
        # K : what i have to offer
        # V : what i actually transmit
        self.Q = nn.Linear(embedding_dim, embedding_dim)
        self.K = nn.Linear(embedding_dim, embedding_dim)
        self.V = nn.Linear(embedding_dim, embedding_dim)

        # at the end we go back from embedding_dim to vocab_size to predict the next character
        self.out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        # we add the character vector and its position vector
        # so the model knows both what and where at the same time
        positions = torch.arange(T)
        entrer = self.embedding(x) + self.position(positions)

        # compute Q, K, V for each character
        Q = self.Q(entrer)
        K = self.K(entrer)
        V = self.V(entrer)

        # Q @ K.T gives a score between every pair of characters
        # we divide by sqrt(32) to avoid the values from exploding
        mask = torch.tril(torch.ones(T, T))
        scores = Q @ K.transpose(-2, -1) / sqrt(64)

        # the mask hides future characters by setting them to -inf
        # after softmax -inf becomes 0 so those positions are ignored
        # without this the model would cheat by looking at the answer in advance
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax turns the scores into probabilities
        # then we multiply by V to get the useful information
        layer = torch.softmax(scores, dim=-1) @ V

        # final layer to go back to vocab_size
        logits = self.out(layer)

        if targets is None:
            loss = None
        else:
            # CrossEntropyLoss expects (B*T, C) and not (B, T, C)
            # so we reshape before computing the loss
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, x, max_new_tokens):
        """
        Generate max_new_tokens new characters given a starting context x.
        """
        for _ in range(max_new_tokens):
            # Crop x to block_size to avoid position embedding index out of range
            # During generation x grows at every step, this keeps it within limits
            x_crop = x[:, -block_size:]
            logits, _ = self(x_crop)

            # Focus only on the last character's prediction
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Sample the next character from the probability distribution
            next_char = torch.multinomial(probs, num_samples=1)

            # Append predicted character to the sequence
            x = torch.cat([x, next_char], dim=1)
        return x

model = Model_language(caracters_nbr, block_size, 64)

# AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
epochs = 10000
for step in range(epochs):
    # Get a batch
    x, y = get_batch(train_data, batch_size, block_size)

    # Forward pass
    logits, loss = model(x, y)
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print loss 
    if step % 1000 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")


# Generate text
# Start from a blank characte
context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=200)
print("\n--- Generate Text ---")
print(decode(generated[0].tolist()))
