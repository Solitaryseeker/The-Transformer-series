# The Transformer series - 14
# Feed-Forward Network (FFN)
**What Happens After Attention?**
In the previous days, we explored how attention allows tokens to interact and exchange information.

But after attention produces a context-aware representation, the Transformer needs to transform that information further.

This is the role of the Feed-Forward Network (FFN).

The FFN applies a learned transformation to each token representation independently, while using the same learned parameters for every token position.

## 1. The Basic Idea

The Feed-Forward Network consists of two linear transformations with a nonlinear activation function between them:
```
Attention Output
       │
       ▼
┌───────────────┐
│ Linear Layer  │
│ d_model → d_ff│
└───────────────┘
       │
       ▼
     ReLU
       │
       ▼
┌───────────────┐
│ Linear Layer  │
│ d_ff → d_model│
└───────────────┘
       │
       ▼
   FFN Output
```
So the basic flow is:

Input → Linear → ReLU → Linear → Output

## 2. Mathematical Form

The original Transformer uses:

$$ FFN(x)=ReLU(xW_1+b_1)W_2+b_2 $$

where:

\(x\) = input token representation
\(W_1\) = first linear-layer weights
\(b_1\) = first bias
\(W_2\) = second linear-layer weights
\(b_2\) = second bias
ReLU = activation function

The first linear layer expands the representation, while the second projects it back to the original dimension.

## 3. Dimension Expansion

In the original Transformer:

$$ d_{model}=512 $$

and:

$$ d_{ff}=2048 $$

Therefore:

512
 ↓
2048
 ↓
512

or:

d_model → d_ff → d_model

The intermediate representation is four times larger than the model dimension.

This expansion provides a larger space for the network to learn nonlinear transformations before returning to the original representation size.

## 4. Why Two Linear Layers?

The first layer expands the representation:

d_model → d_ff

The activation function introduces non-linearity.

The second layer projects the representation back:

d_ff → d_model

Together:

512 → 2048 → 512

This gives the FFN more expressive power than using a single linear transformation.

## 5. Token-Wise Processing

One of the most important properties of the FFN is that it processes each token independently.

Suppose the sequence is:

The | cat | chased | mouse

After attention, we have contextual representations for each token.

The FFN processes them like this:
```
The    → FFN → transformed representation
cat    → FFN → transformed representation
chased → FFN → transformed representation
mouse  → FFN → transformed representation
```
There is no direct interaction between these token positions inside the FFN.

However, the input representations already contain contextual information from the attention mechanism.

## 6. The Same FFN Is Used for Every Token

The same \(W_1\), \(W_2\), \(b_1\), and \(b_2\) parameters are applied to every token position.

Conceptually:
```
Token 1 ──┐
Token 2 ──┤
Token 3 ──┼──► Same FFN parameters
Token 4 ──┤
Token 5 ──┘
```
This is why the FFN is often described as a position-wise feed-forward network.

#  Self-Attention vs FFN

A useful way to understand their different roles is:
```
Self-Attention
      ↓
Communication between tokens
      ↓
FFN
      ↓
Transformation of each token
```
## Self-Attention

Allows tokens to gather information from other tokens.

## Feed-Forward Network

Transforms each token's resulting representation independently.

A simple memory aid:

Attention = communication
FFN = transformation

---
