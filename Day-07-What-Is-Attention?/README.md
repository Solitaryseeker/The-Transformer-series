# The Transformer series - DAY 07
# Day 7 — What Is Attention?

After understanding Tokenization, Embeddings, and Positional Encoding, we now know what the Transformer receives as input and how it represents token order.

But there is still an important question:

**How does a Transformer know which tokens are related to each other?**

This is where Attention comes in.

##cWhat is Attention?

Attention allows a token to look at other tokens in the sequence and determine how important they are for understanding the current context.

Consider:

**The cat sat on the mat because it was tired.**

When the model processes the word "it", it needs context from the other words to understand what "it" refers to.

The model therefore doesn't treat every token as equally important. It learns relationships between tokens and assigns different levels of importance.

A simple way to think about Attention is:
```
Current Token
     ↓
Look at other tokens
     ↓
Determine their importance
     ↓
Use important information
     ↓
Better understanding of context
```
A Simple Example

Consider:**The cat chased the mouse.**

When processing "chased", different tokens can provide different contextual information.
```
The     cat     chased     the     mouse
 ↓       ↓         ↓         ↓        ↓
 low    high     target     low     high
```
The model can learn that "cat" and "mouse" are particularly relevant to understanding the relationship represented by "chased".

This is the intuition behind Attention.

## Attention Is About Relationships

Without Attention, a model would have difficulty directly connecting information across different positions in a sequence.

Attention creates a mechanism for learning relationships such as:
```
Token A ─────────→ Token B
        relationship
```
the model can learn relationships between:
```
cat  ↔ chased
cat  ↔ mouse
chased ↔ mouse
```
The strength of these relationships is represented through attention weights.

# A Simple Attention Visualization 
![]()

## What Does the Attention Matrix Represent?

If the current token is: **chased**
the corresponding row tells us how much attention it gives to:
```
The
cat
chased
the
mouse
```

# Attention ≠ Self-Attention

At this stage, we are focusing on the general idea of Attention.

In a Transformer, the commonly used mechanism is Self-Attention, where tokens in the same sequence attend to one another.

We will build toward that step by step.

The next important question is:

How does the Transformer calculate these attention relationships?

To answer that, we need three concepts:
```
Query (Q)
Key   (K)
Value (V)
```
These three components form the foundation for understanding the mathematics behind Transformer Attention.

