# The Transformer Series — Day 11
# Multi-Head Attention

In the previous sections, we explored Attention, Query-Key-Value (QKV), Scaled Dot-Product Attention, and Self-Attention.

Now we move to one of the core components of the Transformer architecture:

**Multi-Head Attention (MHA)**

The main idea is simple: instead of using only one attention mechanism, the Transformer uses multiple attention heads in parallel.

## 1. Why Multiple Attention Heads?

Consider the sentence: **The cat chased the mouse.**

Self-Attention allows every token to look at the other tokens.

However, a single attention mechanism has only one learned projection space.

Multi-Head Attention creates several different attention mechanisms, allowing the model to learn different patterns from the same input.

Conceptually:
```
                 Input Sequence
                       │
             ┌─────────┴─────────┐
             │                   │
          Head 1              Head 2       ... Head h
             │                   │
      Self-Attention       Self-Attention
             │                   │
             └─────────┬─────────┘
                       ↓
                  Concatenate
                       ↓
                Linear Projection
                       ↓
                     Output
```
The heads work in parallel.

## 2. From One Head to Multiple Heads

A single attention head performs:

$$ head_i = Attention(Q_i,K_i,V_i) $$

where:

$$ Attention(Q_i,K_i,V_i) = softmax \left( \frac{Q_iK_i^T}{\sqrt{d_k}} \right)V_i $$

With multiple heads:

$$ head_1, head_2, \ldots, head_h $$

are calculated independently.

Their outputs are then concatenated:

$$ Concat(head_1,head_2,\ldots,head_h) $$

Finally, a learned linear projection produces the Multi-Head Attention output:

$$ \boxed{ MultiHead(Q,K,V) = Concat(head_1,\ldots,head_h)W^O } $$
## 3. Creating Q, K and V for Each Head

The input representation is projected separately for each head.

For head \(i\):

$$ Q_i = XW_i^Q $$ $$ K_i = XW_i^K $$ $$ V_i = XW_i^V $$

Then:

$$ head_i = Attention(Q_i,K_i,V_i) $$

So each head has its own learned projection matrices.
```
Input X
   │
   ├──→ Q₁ K₁ V₁ → Attention Head 1
   │
   ├──→ Q₂ K₂ V₂ → Attention Head 2
   │
   ├──→ Q₃ K₃ V₃ → Attention Head 3
   │
   └──→ Qₕ Kₕ Vₕ → Attention Head h
```
## 4. Example

For: **The cat chased the mouse.**

different attention heads process the same sequence using different learned projections.

Conceptually:
```
Input
"The cat chased the mouse"
          │
          ├──────────────┐
          ↓              ↓
       Head 1          Head 2
          │              │
          ↓              ↓
    Attention        Attention
          │              │
          └──────┬───────┘
                 ↓
             Concatenate
                 ↓
          Linear Projection
                 ↓
               Output
```
The important point is that we should not assume that Head 1 always learns one specific linguistic relationship and Head 2 another.

The model learns the useful representations during training.

## 5. Complete Multi-Head Attention Pipeline
```
                    Input X
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
     Head 1          Head 2          Head h
        │              │              │
      Q₁K₁V₁         Q₂K₂V₂         QₕKₕVₕ
        │              │              │
        ↓              ↓              ↓
  Scaled Dot-     Scaled Dot-     Scaled Dot-
  Product         Product         Product
  Attention       Attention       Attention
        │              │              │
        └──────────────┼──────────────┘
                       ↓
                  Concatenate
                       ↓
                Linear Projection
                       ↓
                     Output
```
## 6. Why Concatenate the Heads?

Each attention head produces its own representation.

For example:
```
Head 1 → representation 1
Head 2 → representation 2
Head 3 → representation 3
Head 4 → representation 4
```
These representations are concatenated:
```
Head 1 ─┐
Head 2 ─┤
Head 3 ─┼──→ Concatenate → Linear → Output
Head 4 ─┘
```
The final linear layer mixes information from all the heads into the output representation.

## 7. Dimension Example

Suppose:
```
d_model = 512
number of heads = 8
```
A common configuration uses:

$$ d_k=d_v=\frac{512}{8}=64 $$

Therefore, each head works with a smaller representation:
```
512-dimensional input
        ↓
8 attention heads
        ↓
8 × 64-dimensional representations
        ↓
Concatenate
        ↓
512 dimensions
        ↓
Linear projection
```
This allows the model to perform multiple attention operations without making every individual head operate over the full model dimension.

## Self-Attention vs Multi-Head Attention

It is useful to distinguish these concepts.

Self-Attention

A sequence creates Q, K and V from itself:
```
X → Q
X → K
X → V
```
Then one attention mechanism calculates:

$$ Attention(Q,K,V) $$
## Multi-Head Attention

Multiple independent attention mechanisms are calculated:
```
Head 1 → Attention(Q₁,K₁,V₁)

Head 2 → Attention(Q₂,K₂,V₂)

...

Head h → Attention(Qₕ,Kₕ,Vₕ)
```
Then:
```
Concatenate
     ↓
Linear Projection
```
Therefore:

Multi-Head Attention is essentially multiple attention heads operating in parallel and then combining their outputs.

## The Complete Equation

The complete Multi-Head Attention mechanism can be summarized as:

$$ head_i = Attention(Q_i,K_i,V_i) $$

where:

$$ Attention(Q_i,K_i,V_i) = softmax \left( \frac{Q_iK_i^T}{\sqrt{d_k}} \right)V_i $$

Then:

$$ \boxed{ MultiHead(Q,K,V) = Concat(head_1,\ldots,head_h)W^O } $$

---
![](https://github.com/Solitaryseeker/The-Transformer-series/blob/main/assets/mha.png)

[code](1EcQPpafuHrXJCzDOMq8RNvp7emfBG2CH)

The next question is:

**If a model is generating text one token at a time, should it be allowed to look at future tokens?**
