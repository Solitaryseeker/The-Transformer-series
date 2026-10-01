# The Transformer series - DAY 08
# Query, Key & Value (Q, K, V)
In Day 7, we introduced the intuition behind Attention: a token can look at other tokens and determine which information is important.

Now we need to understand how the Transformer represents this process.

This is where Query (Q), Key (K), and Value (V) come in.
## 1. A Simple Example

Consider the sentence:

**The cat chased the mouse.**
After tokenization and embedding, the Transformer has a representation for every token:

**The     cat     chased     the     mouse**

For each token representation, the Transformer creates three different representations:
```
Query (Q)
Key   (K)
Value (V)
```
These are produced through learned linear transformations of the token representation.

Conceptually:
````
Token Representation
        │
        ├──────→ Query (Q)
        │
        ├──────→ Key (K)
        │
        └──────→ Value (V)
````
So Q, K, and V are not three different tokens. They are three representations derived from the token information.

## 2. What Do Q, K and V Mean?
**Query — Q**

The Query represents what the current token is looking for.

For example, when processing:

**chased**

the Query helps the model determine what information is relevant to understanding "chased".

**Key — K**

The Key represents the information that a token can use for matching.

The Query of "chased" can be compared with the Keys of:
```
The
cat
chased
the
mouse
```
**Value — V**

The Value contains the information that will actually be used once a token is considered relevant.

A simple mental model is:

```
Query → What am I looking for?

Key   → What do I represent?

Value → What information can I provide?
```
## 3. Q, K and V in Action

Let's focus on:

The cat chased the mouse.

Suppose the current token is:
```
chased
```
Its representation produces:
```
Q_chased
K_chased
V_chased
```
The model then compares the Query of "chased" with the Keys of the tokens:
```
Q_chased
    │
    ├────→ K_The
    ├────→ K_cat
    ├────→ K_chased
    ├────→ K_the
    └────→ K_mouse
```
These comparisons produce similarity scores.

Conceptually:
```
Q_chased × K_The      → score
Q_chased × K_cat      → score
Q_chased × K_chased   → score
Q_chased × K_the      → score
Q_chased × K_mouse    → score
```
Higher scores indicate that the corresponding token is more relevant to the current Query.

The resulting scores are later converted into attention weights.

## 4. Using the Values

After determining the attention weights, the model uses them to combine the corresponding Values.
```
Attention Weights
       │
       ▼
 ┌─────┬─────┬─────┬─────┬─────┐
 │ The │ cat │chased│ the │mouse│
 └─────┴─────┴─────┴─────┴─────┘
       │
       ▼
Weighted combination of Values
       │
       ▼
Updated representation
```
This gives the Transformer a context-aware representation of the current token.

The actual mathematical calculation of these scores and weights comes next.

## 5. The QKV Pipeline

The complete idea can be summarized as:
```
Token Representations
        │
        ▼
     Generate
      Q, K, V
        │
        ▼
Compare Query
with all Keys
        │
        ▼
Attention Scores
        │
        ▼
Attention Weights
        │
        ▼
Weighted Sum
of Values
        │
        ▼
Context-Aware
Representation

```
This is the foundation of the attention mechanism.

# 6. Important Detail

At this stage, we have not yet implemented complete Self-Attention.

We have only created: Q  K  V

The next step is to calculate how strongly each Query matches each Key.

That leads to the Scaled Dot-Product Attention equation:

$$ Attention(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right)V $$

We will break this equation down step by step in the next part.

![](https://github.com/Solitaryseeker/The-Transformer-series/blob/main/assets/Query%2C%20Key%20and%20Value%20Explained.png)
