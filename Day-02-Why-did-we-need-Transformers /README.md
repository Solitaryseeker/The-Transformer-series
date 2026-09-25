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

------
The Limitations of Recurrent Processing

One important limitation of recurrent architectures is their sequential nature. Because each step depends on information from previous steps, the computation has an inherent dependency across the sequence. This makes it difficult to fully parallelize the processing of a sequence during training.

Another challenge is learning long-range dependencies. When information needs to travel through many recurrent steps, the model can have difficulty preserving the information that is important for a later prediction.

RNNs can also experience optimization problems associated with vanishing and exploding gradients. LSTMs and GRUs introduced gating mechanisms that help address the difficulty of learning longer-term dependencies, but they still retain the fundamental recurrent structure.

Therefore, the problem was not simply that RNNs and LSTMs were unable to process sequences. They were useful sequence models. The larger question was whether sequence relationships could be modeled in a way that provided more direct interactions between different positions and allowed greater parallelism during training.

The Idea of Attention

Attention introduced a different way of thinking about sequence relationships.

Instead of requiring information to be passed only through a chain of recurrent states, an attention mechanism can allow a representation at one position to consider information from other positions in the sequence.

Conceptually, recurrent processing can be represented as:

Token 1 → Token 2 → Token 3 → Token 4 → Token 5

while attention provides the possibility of direct interactions such as:

Token 1 ───────────────► Token 5
Token 2 ───────► Token 4
Token 3 ─────────► Token 5
Token 5 ─────► Token 1

The important idea is that the model can learn which elements of the sequence are relevant to one another.

This provides a different mechanism for representing relationships compared with passing information through a recurrent chain.
