# Transformers from Scratch — Day 16
# Complete Transformer Encoder Layer

## How Do All the Components Work Together?

In the previous parts of this series, we explored Multi-Head Self-Attention, Feed-Forward Networks (FFN), Residual Connections, and Layer Normalization.

Now, we will combine these components to understand a **complete Transformer Encoder Layer**.

A Transformer encoder layer processes an input sequence and produces contextual representations, allowing each token to incorporate information from other tokens in the sequence.

## 1. Encoder Layer Architecture

The original Transformer encoder layer follows this structure:

```text
Input Representations (X)
          │
          ▼
Multi-Head Self-Attention
          │
          ▼
      Add & Norm ◄──── Residual from X
          │
          ▼
Intermediate Representation (H)
          │
          ▼
Feed-Forward Network (FFN)
          │
          ▼
      Add & Norm ◄──── Residual from H
          │
          ▼
Encoder Output (Y)
```

This is the original Transformer's **Post-LayerNorm architecture**.

The encoder layer contains two principal sublayers:

1. Multi-Head Self-Attention
2. Position-wise Feed-Forward Network

Each sublayer is followed by a residual connection and Layer Normalization.

## 2. Multi-Head Self-Attention

The first sublayer allows each token to gather contextual information from the other tokens in the input sequence.

Its mathematical form is:


$$A=\operatorname{MHA}(X,X,X)$$

Here, the input representation \(X\) provides the Queries, Keys, and Values.

Multi-Head Attention uses multiple attention heads to learn different patterns of relationships within the sequence.

## 3. First Add & Norm

The attention output is combined with the original input using a residual connection, followed by Layer Normalization.

$$H=\operatorname{LayerNorm}(X+A)$$

Where:

* \(X\) is the original input.
* \(A\) is the attention output.
* \(H\) is the intermediate representation.

The residual connection provides a direct path for information, while Layer Normalization normalizes the resulting features.

## 4. Feed-Forward Network (FFN)

The intermediate representation is then passed through a position-wise Feed-Forward Network.

The original Transformer uses:

$$ \operatorname{FFN}(H)=\operatorname{ReLU}(HW_1+b_1)W_2+b_2 $$

The FFN transforms each token representation independently, using the same learned parameters at every token position.

It does not directly exchange information between different token positions; that interaction is handled by attention.

## 5. Second Add & Norm

The FFN output is combined with its input and normalized:

$$ \boxed{Y=\operatorname{LayerNorm}(H+\operatorname{FFN}(H))} $$

The result \(Y\) is the output of the encoder layer.

The output retains the sequence length and model dimension of the input, assuming the standard architecture.

## 6. Complete Mathematical Formulation

The original Post-LayerNorm encoder layer can be written as:

**Step 1 — Attention**

$$
A=\operatorname{MHA}(X,X,X)

$$ **Step 2 — First Add & Norm** $$

H=\operatorname{LayerNorm}(X+A)

$$**Step 3 — Feed-Forward Network**$$

F=\operatorname{FFN}(H)

$$**Step 4 — Second Add & Norm**$$\boxed{Y=\operatorname{LayerNorm}(H+F)}$$

Dropout is omitted from these equations for clarity.

## 7. PyTorch Implementation

The following code implements an educational version of the original Transformer encoder layer.

```python
import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        dropout=0.1
    ):
        super().__init__()

        # Multi-Head Self-Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Position-wise Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Self-Attention
        attention_output, _ = self.self_attention(
            x, x, x,
            need_weights=False
        )

        # First residual connection and normalization
        h = self.norm1(
            x + self.dropout1(attention_output)
        )

        # Feed-Forward Network
        ffn_output = self.ffn(h)

        # Second residual connection and normalization
        y = self.norm2(
            h + self.dropout2(ffn_output)
        )

        return y
```

## 8. Testing the Encoder Layer

```python
import torch

# Configuration
batch_size = 2
seq_len = 10
d_model = 512

# Random input representations
x = torch.randn(
    batch_size,
    seq_len,
    d_model
)

# Initialize encoder layer
encoder = TransformerEncoderLayer(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1
)

# Evaluation mode for a reproducible inference-style test
encoder.eval()

with torch.no_grad():
    output = encoder(x)

print("Input shape :", x.shape)
print("Output shape:", output.shape)
```

Expected output:

```text
Input shape : torch.Size([2, 10, 512])
Output shape: torch.Size([2, 10, 512])
```

The dimensions represent:

* `2`: batch size
* `10`: sequence length
* `512`: model dimension

The encoder layer preserves these dimensions while transforming the token representations.

## 9. Where Are Transformer Encoder Layers Used?

### BERT

BERT uses stacked Transformer encoder layers to build contextual representations for language understanding tasks, such as text classification and question answering.

### Original Transformer

The original Transformer uses an encoder stack to process the input sequence and provide representations to its decoder.

### GPT

GPT primarily uses decoder-only Transformer blocks with causal masking. It does not use the original Transformer encoder stack.

These models share Transformer concepts, but their architectures and training objectives differ.

## 10. Key Takeaways

A complete Transformer encoder layer combines several components:

* **Multi-Head Self-Attention:** gathers contextual information across tokens.
* **Residual Connections:** provide shortcut paths for information and gradients.
* **Layer Normalization:** normalizes feature representations.
* **Feed-Forward Network:** applies nonlinear transformations independently to each token.

The original Transformer stacks multiple encoder layers to build increasingly contextual representations.

The essential flow is:

```text
Input
  ↓
Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
Feed-Forward Network
  ↓
Add & Norm
  ↓
Output
```

**The key idea:** Attention gathers information, the FFN transforms it, and Add & Norm combines the sublayer output with its input.

---
![](https://github.com/Solitaryseeker/The-Transformer-series/blob/main/assets/encoder.jpg)
