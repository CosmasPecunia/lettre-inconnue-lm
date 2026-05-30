# lettre-inconnue-lm


# Character-Level Language Model

A character level language model built with PyTorch, 
trained on a French text corpus.

# Project Status
 Work in progress

# What it does
- Reads a text file and builds a character-level vocabulary
- Tokenizes the text into integers
- Samples random batches for training
- learns to predict the next character using Self-Attention

# Roadmap
- [x] Tokenizer and vocabulary
- [x] Batch sampler
- [x] Embedding layer
- [x] Language model (Bigram)
- [x] Training loop(10000)
- [x] Text generation
- [x] Self-Attention mechanism
- [ ] Transformer (attention mechanism)
- [ ] Text generation improvement
      
# Results - Bigram Model
Step 0 | Loss: 4.5799
Step 1000 | Loss: 2.3957
Step 2000 | Loss: 2.2309
Step 3000 | Loss: 2.2067
Step 4000 | Loss: 2.1331
Step 5000 | Loss: 2.3100
Step 6000 | Loss: 2.0646
Step 7000 | Loss: 2.0090
Step 8000 | Loss: 2.2489
Step 9000 | Loss: 2.2099

# Generated Text
"1
enciois re la œuntougorirblé. Je 

por ut emoi à sée. Peurn.  tois.  ctheu ja pardite, ce por te enas antu ie fenchauxans ens el helusie flrs, de 
ste tne » umpre t'urse. [6
D'ain hét s, elssi pi"

# What the model learned
- ✅ real French words starting to appear : "Je", "ce", "de"
- ✅ punctuation patterns : commas, dashes, apostrophes
- ✅ the structure of the book : numbers alone on a line = chapter titles
- ✅ references like [6] from the original text
- ❌ still generating some invented words like "enciois", "helusie"
the model now looks at 32 characters at once instead of just 1,
so it starts to understand the structure of the text, not just character pairs
  
# Requirements
- Python 3.x
- PyTorch

# Dataset
Trained on *Lettre d'une inconnue de Stefan Zweig* — a French literary text.

#Note
took a small break this week because of work,
but i'm back and ready to continue toward Multi-Head Attention !

# Author
cosmas kabaso
