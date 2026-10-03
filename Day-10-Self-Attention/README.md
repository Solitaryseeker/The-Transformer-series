# The Transformer Series — Day 10
# Self-Attention: How Every Token Understands the Others

In the previous posts, we built the Attention mechanism step by step:

Attention → Query, Key & Value → Scaled Dot-Product Attention

Now we can bring these concepts together to understand one of the most important mechanisms in the Transformer architecture:

**Self-Attention**

## 1. What Is Self-Attention?

Consider the sentence: **The cat chased the mouse.**

Each token is converted into three representations:
```
The     → Q₁ K₁ V₁
cat     → Q₂ K₂ V₂
chased  → Q₃ K₃ V₃
the     → Q₄ K₄ V₄
mouse   → Q₅ K₅ V₅
```
The important idea is that every token can use information from the other tokens in the same sequence.

For example, when processing: **chased**

the model can look at:
```
The
cat
chased
the
mouse
```
and determine which tokens are useful for understanding the current context.

## 2. Why Is It Called "Self"-Attention?

The name comes from where the Q, K, and V representations originate.

In Self-Attention:
```
Input Sequence
      │
      ├──→ Q
      ├──→ K
      └──→ V
```
All three come from the same input sequence.

Therefore:

The sequence is attending to itself.

This is different from Cross-Attention, where Queries and Keys/Values can come from different sequences. Cross-Attention will be covered later.

## 3. The Self-Attention Process

For every token, the Transformer performs the same basic process.

### Step 1 — Create Q, K and V

Each token representation is projected into:
```
Q = Query
K = Key
V = Value
```
For the entire sequence:
```
X
│
├──→ Q
├──→ K
└──→ V
```
### Step 2 — Compare Queries with Keys

The model calculates:

$$ QK^T $$

This produces a matrix of similarity scores.

For five tokens:
```
Q shape  → (5, dₖ)
K shape  → (5, dₖ)

QKᵀ      → (5, 5)
```
The resulting matrix tells us how strongly each token relates to every other token.

## 4. The Attention Matrix

For: **The cat chased the mouse**

the attention matrix can be represented as:
```
                 Keys
             The  cat  chased  the  mouse
          ┌───────────────────────────────┐
The       │  •    •      •      •     •   │
cat       │  •    •      •      •     •   │
chased    │  •    •      •      •     •   │
the       │  •    •      •      •     •   │
mouse     │  •    •      •      •     •   │
          └───────────────────────────────┘
             ↑
           Queries
```
Each row represents one Query.

Each column represents a Key.

Therefore, one row answers:

How much attention does this token give to every token in the sequence?

## 5. Scale the Scores

As we saw in Day 9, the similarity scores are scaled by:

$$ \sqrt{d_k} $$

So:

$$ \frac{QK^T}{\sqrt{d_k}} $$

This helps keep the values in a suitable range before applying Softmax.

# 6. Apply Softmax

Next:

$$ \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) $$

Softmax converts the scores into attention weights.

For example, one token might assign weights like:

The       0.10
cat       0.35
chased    0.20
the       0.05
mouse     0.30

These values add up to approximately:

1.0

A larger weight means that the corresponding token contributes more information to the output.

These values are illustrative, not from a trained Transformer.

## 7. Combine the Values

The attention weights are then used to calculate a weighted combination of the Value vectors:

$$ Attention(Q,K,V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)V $$

The result is a new representation for each token.
```
Attention Weights
       │
       ▼
Weighted Values
       │
       ▼
Context-Aware Representation
```
This means the representation of a token is no longer based only on that token.

It now contains information gathered from the context of the entire sequence.

## 8. Complete Self-Attention Pipeline

The entire process can be visualized as:
```
Input Sequence
"The cat chased the mouse"
          │
          ▼
     Token Representations
          │
          ▼
       Generate Q, K, V
          │
          ▼
          QKᵀ
          │
          ▼
     Scale by √dₖ
          │
          ▼
        Softmax
          │
          ▼
    Attention Weights
          │
          ▼
    Weighted Sum of V
          │
          ▼
Context-Aware Representations
```
The important difference from processing tokens independently is that each output representation now contains contextual information.

## Why Self-Attention Is Important

Self-Attention allows a Transformer to capture relationships between tokens even when they are far apart in a sequence.

For example: **The cat chased the mouse because it was hungry.**

Understanding the meaning of "it" requires information from the surrounding context.

Self-Attention gives the model a mechanism for connecting information across the sequence.

## Attention vs Self-Attention

It is useful to distinguish the general concept from the specific Transformer mechanism.

|Concept	|Main idea|
|---|---|
|Attention|	A mechanism for weighting relevant information|
|Self-Attention	|Attention where Q, K and V come from the same sequence|
|Cross-Attention	|Attention where the information comes from another sequence|

For example:

Self-Attention:
```
Sequence → Q
Sequence → K
Sequence → V
```
Whereas Cross-Attention will look more like:
```
Sequence A → Q

Sequence B → K
Sequence B → V
```
We will study Cross-Attention later.

# The Core Formula

The complete Self-Attention calculation is:

$$ \boxed{ \text{SelfAttention}(X) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)V } $$

where:

$$ Q=XW_Q $$ $$ K=XW_K $$ $$ V=XW_V $$

Therefore:
```
X
│
├──→ XWQ → Q
├──→ XWK → K
└──→ XWV → V
```
Then:

$$ Q,K,V \rightarrow \text{Attention} \rightarrow \text{Context-Aware Representations} $$
Key Takeaway

Self-Attention allows every token to look at every other token in the same sequence and gather the information needed to understand its context.

The complete process is:
```
Input
  ↓
Q, K, V
  ↓
QKᵀ
  ↓
Scale
  ↓
Softmax
  ↓
Attention Weights
  ↓
Weighted Sum of V
  ↓
Context-Aware Representations
```
This is the core mechanism that allows Transformers to model relationships between tokens.
![](https://github.com/Solitaryseeker/The-Transformer-series/blob/main/assets/Self-Attention%20Explained_%20Transformer%20Infographic.png)
