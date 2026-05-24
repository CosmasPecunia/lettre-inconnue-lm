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

class Model_language(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # The embedding table IS the model
        # Each row = scores (logits) for the next character
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        # x shape: (B, T)
        # logits shape: (B, T, vocab_size)
        logits = self.embedding(x)

        if targets is None:
            loss = None
        else:
            # Reshape for CrossEntropyLoss
            B, T, C = logits.shape
            logits = logits.view(B * T, C)   # (B*T, vocab_size)
            targets = targets.view(B * T)     # (B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, x, max_new_tokens):
        """
        Generate max_new_tokens new characters given a starting context x.
        """
        for _ in range(max_new_tokens):
            # Get predictions
            logits, _ = self(x)
            # Focus only on the last character
            logits = logits[:, -1, :]            # (B, caracters_nbr)
            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)    # (B, caracters_nbr)
            # Sample the next character
            next_char = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append to the sequence
            x = torch.cat([x, next_char], dim=1)                 # (B, T+1)
        return x


model = Model_language(caracters_nbr)

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
