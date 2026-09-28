# The Transformer series - DAY 05
#   How Does a Transformer Read Human Language?

## Introduction

A Transformer does not directly read human language the way we do.

If we give a model a sentence such as:

> "The cat chased the mouse."

the model cannot process the raw sentence directly as human-readable text.

Before the Transformer can work with the sentence, the text needs to be converted into a numerical representation that a neural network can process.

The basic journey is:
![](https://github.com/Solitaryseeker/The-Transformer-series/blob/main/assets/read.png)

This is the first step in understanding how text enters a Transformer.

## From Text to Tokens

The first step is Tokenization.

Tokenization converts text into smaller units called tokens.

For example:
```text
"The cat chased the mouse."
```
can be represented as:
```text
["the", "cat", "chased", "the", "mouse"]
```
The exact tokens depend on the tokenizer and vocabulary used by a particular model.

A token does not always have to be a complete word. Modern language models can also use subwords, punctuation, spaces, or other token units.

The important idea is:

**Tokenization converts human-readable text into units that the model can represent numerically.**

## Token IDs

The Transformer still cannot directly process token strings such as:
```text
"the"
"cat"
"chased"
```
Each token is therefore mapped to an integer called a Token ID.

For example:
```text
"the"     → 1437
"cat"     → 5389
"chased"  → 7234
"the"     → 1437
"mouse"   → 4321
```
These numbers are identifiers from the model's vocabulary.

It is important to understand that the ID itself does not contain the meaning of the word.

For example:
```text
"cat" → 5389
```
does not mean that the number 5389 mathematically represents the meaning of "cat".

It is simply an identifier used to look up the corresponding learned representation.

From Token IDs to Embeddings

The next step is Embedding.

Each token ID is associated with a numerical vector.

For example:
```text
"cat" → 5389
          ↓
[0.8, 0.5, 0.1, 0.9, 0.7, 0.2]
```
This vector is called a token embedding.

Instead of representing a token as a single integer, the model represents it using a vector with many dimensions.

Conceptually:
```text
Token ID
   ↓
Embedding Lookup
   ↓
Vector Representation
```
The embedding layer can therefore be thought of as a learned lookup table that maps token IDs to vectors.

# Why Do We Need Embeddings?

A token ID such as:
```text
5389
```
is only an identifier.

A vector representation gives the neural network a numerical space in which it can learn useful relationships between tokens.

For example:
```text
cat    → [ ... ]
dog    → [ ... ]
car    → [ ... ]
```
During training, the embedding representations are learned as part of the model.

The model can therefore operate on numerical vectors rather than raw text.

# One Important Detail

Tokenization and embeddings are related, but they are not the same thing.

Tokenization answers:

**How should the text be divided into tokens?**

Token IDs answer:

**Which vocabulary entry represents each token?**

Embeddings answer:

**What numerical vector represents each token?**

So:
```text
Tokenization → Token IDs → Embeddings
```
are three connected but different steps.

# What Happens Next?

We now have numerical representations of the tokens.

But there is a new problem.

Consider:
```text
"The cat chased the mouse."
```
The Transformer needs to know not only what the tokens are, but also where they occur in the sequence.

For example:
```text
the   → position 1
cat   → position 2
chased → position 3
the   → position 4
mouse → position 5
```
How does a Transformer represent this positional information?

That leads to the next concept:

# Positional Encoding

Before we study how tokens interact with each other through Attention, we first need to understand how the Transformer knows their positions.

Once the text has become numerical representations, we can begin asking the more interesting question:

**How does the Transformer understand the relationships between these tokens?**

That is where Attention begins.
