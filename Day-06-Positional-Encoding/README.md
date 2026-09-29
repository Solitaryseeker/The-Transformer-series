# The Transformer series - DAY 06
#  Positional Encoding

## Introduction

A Transformer can process tokens in parallel, but this creates an important question:

**How does the Transformer know the position of each token in a sequence?**

Consider these two sentences:

> "The cat chased the mouse."

> "The mouse chased the cat."

They contain the same words, but changing their positions changes the meaning.

This is why positional information is important in a Transformer.

---

## Why Do We Need Positional Encoding?

Unlike sequential models such as RNNs, the original Transformer does not process tokens one by one.

The Transformer processes the sequence in parallel.

Therefore, the model needs additional information that tells it where each token occurs in the sequence.

For example:

```
The      → Position 0
cat      → Position 1
chased   → Position 2
the      → Position 3
mouse    → Position 4
```
Positional Encoding provides this information.

## The Basic Idea

A Transformer first converts tokens into embedding vectors.

Then positional information is added to those embeddings.
```
Token
  ↓
Tokenization
  ↓
Token ID
  ↓
Token Embedding
  +
Positional Encoding
  ↓
Input Representation
  ↓
Transformer
```
The embedding represents what the token is, while positional encoding provides information about where the token is.

## Original Transformer Positional Encoding

In the original Transformer paper, positional encodings are generated using sine and cosine functions with different frequencies.

For even dimensions:
```
PE(pos, 2i) =
sin(pos / 10000^(2i / d_model))
```
For odd dimensions:
```
PE(pos, 2i + 1) =
cos(pos / 10000^(2i / d_model))
```
Where:
pos      → position of the token

i        → dimension index

d_model  → embedding dimension

## Why Sine and Cosine?

The sine and cosine functions create different patterns for different positions and dimensions.

Conceptually:
```
Position 0 → [sin(...), cos(...), sin(...), cos(...), ...]

Position 1 → [sin(...), cos(...), sin(...), cos(...), ...]

Position 2 → [sin(...), cos(...), sin(...), cos(...), ...]
```

Each position therefore receives a distinct numerical pattern.

These patterns are added to the token embeddings.

# What Does Positional Encoding Actually Provide?

Positional Encoding helps the Transformer distinguish between tokens that occur at different positions.

Without positional information, the model would have difficulty distinguishing sequences where the same tokens appear in different orders.
For example:
```
The cat chased the mouse
```
and
```
The mouse chased the cat
```
have the same set of words but different arrangements.

Position provides information about this arrangement.
## Important Distinction

Positional Encoding does not tell the model the meaning of a word.

For example:
```
Token Embedding
→ information about the token

Positional Encoding
→ information about its position
```
They work together:
```
Token Meaning
      +
Token Position
      ↓
Input Representation
```
![]()
