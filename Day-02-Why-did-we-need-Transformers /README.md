# Transformers from Scratch — Day 02
## Why did we need Transformers in the first place? 


## Introduction

After understanding what the term "Transformer" means in Day 01, the next question is more fundamental: **Why did we need the Transformer architecture in the first place?**

Before Transformers became widely used, sequence-based problems were commonly addressed using recurrent architectures such as Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Gated Recurrent Units (GRUs). These architectures provided an effective way to process sequential information, but they also introduced important limitations when dealing with long sequences and large-scale training.

The development of the Transformer can therefore be understood as part of the evolution of sequence modeling: from recurrent processing, toward attention-based processing, and eventually toward architectures capable of efficiently modeling relationships across an entire sequence.

---

## The Challenge of Sequence Data

Many real-world datasets contain information where the order of elements matters. Natural language is one example, because the meaning of a word can depend on other words in the sentence. Speech is another example because the interpretation of one sound can depend on surrounding sounds. Time-series data also contains sequential relationships because observations are ordered in time.

Consider the sentence:

> "The scientist who worked on the project published the paper."

To understand this sentence, a model needs to learn relationships between different words. The relationship between "scientist" and "published" is important even though several other words occur between them.

This creates a fundamental challenge for sequence models: **how can a model effectively capture relationships between elements that may be close together or far apart in a sequence?**

---

## Before Transformers: Recurrent Neural Networks

RNNs process a sequence one element at a time. At each step, the model receives the current input together with information carried from the previous step.

Conceptually, the information flow can be represented as:

```text
Token 1 → Token 2 → Token 3 → Token 4 → Token 5
   ↓         ↓         ↓         ↓         ↓
 Hidden    Hidden    Hidden    Hidden    Hidden
 State     State     State     State     State
```
---
## The Limitations of Recurrent Processing

One important limitation of recurrent architectures is their sequential nature. Because each step depends on information from previous steps, the computation has an inherent dependency across the sequence. This makes it difficult to fully parallelize the processing of a sequence during training.

Another challenge is learning long-range dependencies. When information needs to travel through many recurrent steps, the model can have difficulty preserving the information that is important for a later prediction.

RNNs can also experience optimization problems associated with vanishing and exploding gradients. LSTMs and GRUs introduced gating mechanisms that help address the difficulty of learning longer-term dependencies, but they still retain the fundamental recurrent structure.

Therefore, the problem was not simply that RNNs and LSTMs were unable to process sequences. They were useful sequence models. The larger question was whether sequence relationships could be modeled in a way that provided more direct interactions between different positions and allowed greater parallelism during training.

---
## The Idea of Attention

Attention introduced a different way of thinking about sequence relationships.

Instead of requiring information to be passed only through a chain of recurrent states, an attention mechanism can allow a representation at one position to consider information from other positions in the sequence.

Conceptually, recurrent processing can be represented as:
```text
Token 1 → Token 2 → Token 3 → Token 4 → Token 5
```
while attention provides the possibility of direct interactions such as:
```text
Token 1 ───────────────► Token 5
Token 2 ───────► Token 4
Token 3 ─────────► Token 5
Token 5 ─────► Token 1
```
The important idea is that the model can learn which elements of the sequence are relevant to one another.

This provides a different mechanism for representing relationships compared with passing information through a recurrent chain.

# From Attention to the Transformer

The Transformer architecture was introduced by Vaswani et al. in the 2017 research paper "Attention Is All You Need."

The architecture was designed around attention mechanisms and introduced an approach to sequence-to-sequence modeling that removed the need for recurrence from the core architecture.

Instead of relying on recurrent processing, the Transformer uses attention-based interactions together with other components that form a complete neural architecture.

The original Transformer contains an Encoder and a Decoder. The Encoder processes the input representation, while the Decoder generates the output representation. Within these components, attention mechanisms and feed-forward transformations are combined with residual connections and normalization.

# Why Parallel Processing Matters

One of the important characteristics of the Transformer is its ability to process the elements of a sequence more efficiently in parallel during training than recurrent architectures.

In a recurrent model, the computation has a sequential dependency:
```text
Step 1 → Step 2 → Step 3 → Step 4 → Step 5
```
The Transformer instead performs attention operations over representations of the sequence using matrix-based computation:
```text
Token 1 ─┐
Token 2 ─┤
Token 3 ─┼──► Attention Computation
Token 4 ─┤
Token 5 ─┘
```
This property became particularly important as models and datasets grew larger.

However, parallel computation does not mean that Transformers have no computational challenges. Standard self-attention has a computational and memory cost that grows with the length of the sequence, which later became an important research problem.

---
# Why Long-Range Relationships Matter

Consider a sequence containing hundreds or thousands of elements. Important information may occur at positions that are far apart.

A useful sequence model therefore needs to represent relationships across different positions rather than focusing only on neighboring elements.

Attention provides a mechanism through which each position can interact with other positions. This makes it possible for the model to learn relationships that may span large portions of a sequence.

This idea is one of the reasons attention became such an important component of modern sequence modeling.

---
# The Transformer Architecture

The Transformer combines several components rather than relying on attention alone.

The major components include input embeddings, positional information, self-attention, multi-head attention, feed-forward networks, residual connections, and layer normalization. The original architecture organizes these components into Encoder and Decoder stacks.

---
# Where Are Transformers Used?

The Transformer architecture was originally introduced for sequence-to-sequence modeling, but its underlying ideas have since been adapted to many areas of Artificial Intelligence.

In Natural Language Processing, Transformer-based architectures are used for tasks such as language understanding, machine translation, text generation, summarization, and question answering.

Transformer architectures also became the foundation for many Large Language Models. Different models use different Transformer configurations and training objectives, but the underlying attention-based architecture remains an important foundation.

Transformers have also been adapted for Computer Vision. Vision Transformer approaches represent images in a form that allows Transformer-style processing to be applied to image data.

Another important application area is time-series forecasting. Transformer-based models have been explored for weather forecasting, energy forecasting, financial data, demand forecasting, sensor data, and other temporal problems.

The architecture has also been applied to speech processing and multimodal learning, where models need to work with different types of information such as text, images, audio, or combinations of these modalities.

---
# Why Transformers Became Important

The importance of Transformers comes from more than a single architectural component. The architecture introduced a different way of thinking about sequence modeling.

Instead of relying primarily on recurrent information flow, Transformers use attention-based interactions to model relationships between different positions while enabling highly parallel computation during training.

This combination helped make the architecture highly useful for large-scale Deep Learning systems and provided a foundation for many later developments in AI.

At the same time, Transformers introduced new challenges, including the computational cost of attention for very long sequences. These challenges have become important areas of ongoing research.

---

The transition from recurrent architectures to Transformers can be summarized conceptually as:

![](https://github.com/Solitaryseeker/The-Transformer-series/blob/main/assets/why_do.png)

---
# Reference
Vaswani et al. — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

This paper introduced the original Transformer architecture and is the primary reference for this part of the series.
