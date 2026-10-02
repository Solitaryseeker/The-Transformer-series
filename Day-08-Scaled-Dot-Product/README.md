# The Transformer Series — Day 9
# Scaled Dot-Product Attention

In Day 8, we introduced Query (Q), Key (K), and Value (V).

Now we can answer the next question:

**How does the Transformer calculate how strongly a Query matches each Key?**

The answer is Scaled Dot-Product Attention.

It is the mathematical operation that turns Q, K, and V into a context-aware representation.

## 1. The Core Formula

The complete equation is:

$$ \text{Attention}(Q,K,V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)V $$

Although the equation looks complicated, it can be understood as a sequence of simple operations:
```
Q × Kᵀ
   ↓
Similarity Scores
   ↓
Divide by √dₖ
   ↓
Softmax
   ↓
Attention Weights
   ↓
Weighted Sum of V
   ↓
Context-Aware Representation
```
## 2. Example

Consider: **The cat chased the mouse.**

Suppose we are processing the token: **chased**

The Transformer has: **Q_chased**

and Keys for every token:
```
K_The
K_cat
K_chased
K_the
K_mouse
```
The Query is compared with all of these Keys.

Conceptually:
```
Q_chased × K_The      → score
Q_chased × K_cat      → score
Q_chased × K_chased   → score
Q_chased × K_the      → score
Q_chased × K_mouse    → score
```
These are the raw attention scores.

## 3. Step 1 — Dot Product

The first operation is:

$$ QK^T $$

The dot product measures how well the Query and Key vectors align.

For a single Query: **Q = [q₁, q₂, q₃, q₄]**

and a Key: **K = [k₁, k₂, k₃, k₄]**

their dot product is:

$$ Q\cdot K = q_1k_1+q_2k_2+q_3k_3+q_4k_4 $$

For all tokens, this becomes the matrix operation:

$$ QK^T $$
## 4. Step 2 — Scaling

The raw scores are divided by:

$$ \sqrt{d_k} $$

where \(d_k\) is the dimension of the Key vectors.

So:

$$ \frac{QK^T}{\sqrt{d_k}} $$

**Why scale?**

As the dimensionality of the vectors increases, dot products can become large.

Large values passed into Softmax can produce extremely concentrated probabilities and make optimization more difficult.

Scaling keeps the scores in a more suitable range.
```
Raw Scores
    ↓
Divide by √dₖ
    ↓
Scaled Scores
```
## 5. Step 3 — Softmax

The scaled scores are passed through Softmax.

$$ \text{softmax}(x_i) = \frac{e^{x_i}} {\sum_j e^{x_j}} $$

Softmax converts the scores into attention weights.

For example:
```
Scaled Scores:

The       0.12
cat       0.29
chased    0.40
the       0.16
mouse     0.06
```
After Softmax, we obtain values that behave like weights:
```
The       0.14
cat       0.24
chased    0.37
the       0.18
mouse     0.07
```
The weights sum approximately to: **1.0**

A larger weight means that the corresponding Value contributes more strongly to the output.

Note: These numbers are only illustrative. They are not outputs from a trained Transformer.

6. Step 4 — Weighted Sum of Values

Now the attention weights are multiplied by the corresponding Value vectors.

$$ \text{Attention Weights} \times V $$

Conceptually:
```
0.14 × V_The
+
0.24 × V_cat
+
0.37 × V_chased
+
0.18 × V_the
+
0.07 × V_mouse
```
The result is a new representation containing information gathered from the entire sequence.
```
Attention Weights
        ↓
Weighted Values
        ↓
Context-Aware Representation
```
## 7. Complete Process

The entire operation can therefore be understood as:
```
             Query (Q)
                 │
                 ▼
           ┌──────────┐
           │   QKᵀ    │
           └────┬─────┘
                │
                ▼
          Divide by √dₖ
                │
                ▼
             Softmax
                │
                ▼
       Attention Weights
                │
                ▼
             × Values
                │
                ▼
     Context-Aware Output
```
Or mathematically:

$$ \boxed{ \text{Attention}(Q,K,V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)V } $$

## 8. Why Is It Called "Scaled Dot-Product Attention"?

The name comes directly from the operation:

Dot-Product
$$ QK^T $$

We use the dot product to measure similarity between Queries and Keys.

Scaled
$$ \frac{QK^T}{\sqrt{d_k}} $$

We divide by \(\sqrt{d_k}\) to control the magnitude of the scores.

Attention
$$ \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

The resulting weights determine how information from the Values is combined.

