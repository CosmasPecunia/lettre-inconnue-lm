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
- Maps characters to 32-dimensional embedding vectors

# Roadmap
- [x] Tokenizer and vocabulary
- [x] Batch sampler
- [x] Embedding layer
- [x] Language model (Bigram)
- [x] Training loop(10000)
- [x] Text generation
- [ ] Transformer (attention mechanism)
- [ ] Text generation improvement
      
# Results - Bigram Model
Step 0 | Loss: 5.2018
Step 1000 | Loss: 4.4833
Step 2000 | Loss: 3.8612
Step 3000 | Loss: 3.1070
Step 4000 | Loss: 3.0834
Step 5000 | Loss: 2.7855
Step 6000 | Loss: 2.3579
Step 7000 | Loss: 2.3062
Step 8000 | Loss: 2.4342
Step 9000 | Loss: 2.2533

# Generated Text
"aîLre. –Bwjeschant l'e –SQô[Juis mai N’'on 
de a quence, que d'et jos, Cêhitos qu 
hu ymais londé ce c 
M4xüœDV-hinempasin.é en. di jobl mman maire pau coy’: mes trais. j'et)V»dace mêM)Esie le leguina"

# What the model learned
Even with a simple Bigram model, some French patterns emerged:
- ✅ The model learned that **"Q" is almost always followed by "U"** 
- ✅ Basic punctuation patterns (spaces, commas)
- ❌ Cannot generate real words yet (only sees 1 character at a time)
  
# Requirements
- Python 3.x
- PyTorch

# Dataset
Trained on *Lettre d'une inconnue de Stefan Zweig* — a French literary text.

# Author
cosmas kabaso
