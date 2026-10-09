# Transformers from Scratch — Day 15
# Residual Connections & Layer Normalization (Add & Norm)

## Introduction

In the previous parts of this series, we explored Self-Attention, Multi-Head Attention, Masked Self-Attention, Cross-Attention, and the Feed-Forward Network (FFN).

Now, we will study two important components of the Transformer architecture: **Residual Connections** and **Layer Normalization**.

Residual connections provide a shortcut for information to flow around a sublayer, while Layer Normalization normalizes the features of a representation.

Together, they form the **Add & Norm** operation used in the original Transformer architecture.

## 1. Residual Connections — Add

A residual connection adds the input of a sublayer to its output.

### Formula

$$
\boxed{y=x+F(x)}
$$

Where:

* \(x\) is the original input.
* \(F(x)\) is the output produced by the sublayer.
* \(y\) is the combined representation.

### How It Works

```text
              Input (x)
              /       \
             ↓         │
          Sublayer     │
             ↓         │
            F(x)       │
             \         /
               Add (+)
                 ↓
               x + F(x)
```

Instead of relying only on the sublayer's output, the model adds the original input to the transformed representation.

### Why Is It Important?

Residual connections provide a direct path for information and can help gradients propagate through deeper networks during training.

## 2. Layer Normalization — Norm

Layer Normalization normalizes features within a token representation using statistics calculated across the normalized feature dimensions.

### Normalization Formula

$$
\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

The normalized values are then scaled and shifted using learnable parameters:

$$
\boxed{y_i=\gamma\hat{x}_i+\beta}
$$

Where:

* \(x_i\): input feature.
* \(\mu\): mean of the normalized features.
* \(\sigma^2\): variance of the normalized features.
* \(\epsilon\): small constant for numerical stability.
* \(\gamma\): learnable scale parameter.
* \(\beta\): learnable shift parameter.

### Why Is It Important?

Layer Normalization helps keep activation values well-behaved during training and makes the model's optimization process more manageable.

## 3. How Add & Norm Works

In the original Transformer, the Add & Norm pattern follows this sequence:

```text
Input (x)
    │
    ▼
  Sublayer
    │
    ▼
 Sublayer Output F(x)
    │
    ▼
 Add: x + F(x)
    │
    ▼
 Layer Normalization
    │
    ▼
  Output
```

### Combined Formula

$$
\boxed{y=\operatorname{LayerNorm}(x+F(x))}
$$

The residual connection first combines the original input with the sublayer output. Layer Normalization then normalizes the resulting representation.

This is the **Post-LayerNorm arrangement** used in the original Transformer.

## 4. PyTorch Implementation

The following implementation demonstrates the original Transformer's Add & Norm operation.

```python
import torch
import torch.nn as nn


class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        # Residual connection
        combined = x + sublayer_output

        # Layer normalization
        output = self.layer_norm(combined)

        return output


# Example
batch_size = 2
seq_len = 5
d_model = 512

x = torch.randn(batch_size, seq_len, d_model)
sublayer_output = torch.randn(batch_size, seq_len, d_model)

add_norm = AddNorm(d_model)

output = add_norm(x, sublayer_output)

print("Input shape:", x.shape)
print("Sublayer output shape:", sublayer_output.shape)
print("Final output shape:", output.shape)
```

Expected output:

```text
Input shape: torch.Size([2, 5, 512])
Sublayer output shape: torch.Size([2, 5, 512])
Final output shape: torch.Size([2, 5, 512])
```

The input and sublayer output must have compatible shapes for the residual addition. In this example, the final representation retains the same shape.

## 5. Understanding the Difference

| Component           | Main Function                                  |
| ------------------- | ---------------------------------------------- |
| Residual Connection | Adds the original input to the sublayer output |
| Layer Normalization | Normalizes the resulting features              |
| Add & Norm          | Combines residual addition and normalization   |

A simple way to remember:

**Add = Information Shortcut**

**Norm = Feature Normalization**

## 6. Where Is Add & Norm Used?

In the original Transformer architecture, this pattern is used around the attention and feed-forward sublayers.

For example, the encoder applies it around:

* Multi-Head Self-Attention.
* Feed-Forward Network.

The decoder applies it around its masked self-attention, cross-attention, and feed-forward sublayers.

This README focuses on the Add & Norm mechanism itself rather than assembling the complete encoder or decoder layer.

## Key Takeaway

Residual Connections and Layer Normalization have different but complementary roles.

The residual connection combines the original input with the sublayer's output. Layer Normalization then normalizes the combined representation.

$$
\boxed{\operatorname{AddNorm}(x,F(x))
=\operatorname{LayerNorm}(x+F(x))}
$$

Understanding these operations helps explain how the original Transformer combines its attention and feed-forward components into deeper networks.

---

![](https://github.com/Solitaryseeker/The-Transformer-series/blob/main/assets/Residual%20Connections%20and%20Layer%20Normalisation%20Infographic.png)
