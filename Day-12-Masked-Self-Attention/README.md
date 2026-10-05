# The Transformer series - 12
# Masked Self-Attention
**Why Can't the Model Look Ahead?**

Masked Self-Attention is a modified form of self-attention used when a Transformer generates text from left to right.

During generation, the model must predict the next token using only the tokens that are already available. It should not see future tokens, because that would give the model information about the answer in advance.

For example: **The cat chased the mouse**

When the model is processing: **The cat chased the ...**

it can use:

**The → cat → chased**

but it must not use:

**the → mouse**

This is achieved using a causal mask.

## 1. The Causal Mask

For the sequence:

The | cat | chased | the | mouse

the allowed attention pattern is:
```
        The  cat  chased  the  mouse
The      ✓    ✗     ✗      ✗     ✗
cat      ✓    ✓     ✗      ✗     ✗
chased   ✓    ✓     ✓      ✗     ✗
the      ✓    ✓     ✓      ✓     ✗
mouse    ✓    ✓     ✓      ✓     ✓
```
Each token can attend to itself and tokens before it, but not tokens after it.

This creates a lower-triangular attention pattern.

## 2. How Masked Self-Attention Works

The basic attention calculation is:
```
QKᵀ
 ↓
Scale
 ↓
Apply Causal Mask
 ↓
Softmax
 ↓
Attention Weights
 ↓
Weighted Sum of V
```
The mathematical form is:

$$ \text{MaskedAttention}(Q,K,V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} + M \right)V $$

where:

Q = Query
K = Key
V = Value
\(d_k\) = dimension of the Key vectors
M = causal mask that blocks future positions

The masked positions receive effectively \(-\infty\) before Softmax, so their attention probability becomes approximately zero.

![](https://github.com/Solitaryseeker/The-Transformer-series/blob/main/assets/Masked%20Self-Attention%20Infographic.png)

[code]()

# Self-Attention vs Masked Self-Attention
|Self-Attention|	Masked Self-Attention|
|---|---|
|Tokens can attend to other tokens	|Tokens cannot attend to future tokens|
|Future positions may be visible	| Future positions are blocked |
|Useful for bidirectional/contextual processing	|Essential for autoregressive generation|
|No causal mask required |	Uses a causal mask|

# Key Takeaway

Masked Self-Attention = Self-Attention + Causal Mask

The attention mechanism itself remains the same, but the mask controls which tokens are allowed to communicate.

This simple restriction is what allows a Transformer decoder to generate text one token at a time without looking ahead.
