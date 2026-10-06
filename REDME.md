# The Transformer series - 13 
# Cross-Attention
# How Does the Decoder Look at the Encoder?

In the previous part, we learned about Masked Self-Attention, where the decoder can attend only to the current and previous tokens.

But this creates an important question:

**How does the decoder get information from the input sequence processed by the encoder?**

The answer is **Cross-Attention.**

Cross-Attention allows the decoder to selectively retrieve useful information from the encoder output while generating the target sequence.

## 1. The Main Idea

The most important difference between Self-Attention and Cross-Attention is the source of Q, K, and V.

In Self-Attention:
```
Q ← Same Sequence
K ← Same Sequence
V ← Same Sequence
`````
In Cross-Attention:
```

                Encoder Output
                  │       │
                  ↓       ↓
                  K       V

Decoder Output
      │
      ↓
      Q
```
So: **Q comes from the decoder, while K and V come from the encoder.**

## 2. Cross-Attention Flow

The complete process is:
```
Encoder Output
      │
      ├──────────► K
      │
      └──────────► V

Decoder Representation
      │
      └──────────► Q

            QKᵀ
              ↓
            Scale
              ↓
           Softmax
              ↓
      Weighted Sum of V
              ↓
       Decoder Context
```
The decoder uses its Query to determine which parts of the encoder representation are relevant.

## 3. Scaled Dot-Product Attention

Cross-Attention uses the same scaled dot-product attention operation:

$$ Attention(Q,K,V) = Softmax \left( \frac{QK^T}{\sqrt{d_k}} \right)V $$

Where:

Q → Query from the decoder
K → Key from the encoder
V → Value from the encoder
\(d_k\) → dimension of the Key vectors

The formula is the same as regular attention.

The important difference is where Q, K, and V originate.

## 4. Example: Machine Translation

Consider a translation task:
```
Source:
"The cat is sleeping."

        ↓

     Encoder

        ↓

Encoder Representations

        ↓
   ┌──────────────┐
   │ Cross-       │
   │ Attention    │
   └──────────────┘
        ↑
        │
     Decoder

        ↓

Target:
"Le chat dort."
```
The encoder first creates contextual representations of the source sentence.

When the decoder generates the target sequence, its current representation produces Q.

The encoder representations provide K and V.

The attention mechanism then determines how strongly the decoder should use different parts of the encoder output.

## 5. Self-Attention vs Cross-Attention

|Feature |	Self-Attention |	Cross-Attention|
|---|---|---|
|Query (Q)	|Same sequence	|Decoder|
|Key (K) |	Same sequence |	Encoder |
Value (V) |	Same sequence|	Encoder|
Main purpose|	Build context within a sequence |	Access encoder information|
Used in decoder	|Yes	|Yes|

# A simple way to remember it:

Self-Attention

**"Look within my own sequence."**

Cross-Attention

**"Look at the encoder to find useful information."**

# Why No Causal Mask Here?

In the decoder, Masked Self-Attention prevents the decoder from looking at future target tokens.

Cross-Attention is different.

The decoder is allowed to access the encoder's complete output, because the source/input sequence is already available.

So the decoder layer conceptually contains:
```
Decoder Input
     │
     ↓
Masked Self-Attention
     │
     ↓
Cross-Attention
     ↑
     │
Encoder Output
     │
     ↓
Feed-Forward Network
```
This distinction is important when understanding the complete Transformer decoder.

----
![](https://github.com/Solitaryseeker/The-Transformer-series/tree/main/Day-01-What-Do-We-Mean%20by%20-Transformer)

The next question is:

**After attention produces a context-aware representation, what happens to each token?**
That leads to the Feed-Forward Network (FFN).

