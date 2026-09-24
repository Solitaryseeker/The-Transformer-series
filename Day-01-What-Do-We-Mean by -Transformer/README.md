# The Transformer series - 01 

# What Do We Mean by “Transformer”?

When you hear the word **“Transformer”**, what do you think of?

![](https://github.com/Solitaryseeker/The-Transformer-series/blob/main/assets/transformer3.jpeg)

🤖 A transforming robot?
⚡ An electrical transformer?
🧠 Or a Deep Learning architecture?

The interesting part is that all three are called “Transformers” — but they represent completely different concepts.

## Introduction

The word **Transformer** can refer to completely different concepts depending on the field in which it is used. In everyday conversation, it may immediately bring to mind a fictional robot that changes its form. In electrical engineering, a transformer is a physical device used to transfer electrical energy between circuits and change voltage levels. In Artificial Intelligence, however, a Transformer refers to a Deep Learning architecture that uses attention mechanisms to model relationships between elements in a sequence.

Although these concepts share the same word, their purposes, mechanisms, and applications are completely different. This distinction is important because the focus of this series is specifically the **Transformer architecture used in Artificial Intelligence and Deep Learning**.

---

## Three Different Meanings of "Transformer"

### Transformer Robot — Fiction

In popular culture, a Transformer can refer to a fictional robot capable of changing its physical form. These characters are generally associated with shape-shifting robots, fictional stories, and entertainment. This meaning has no technical relationship with the Transformer architecture used in Artificial Intelligence, even though the same word is used.

### Electrical Transformer — Power Systems

In electrical engineering, a transformer is a device that transfers electrical energy between circuits through electromagnetic induction. It can be used to increase or decrease voltage levels depending on the requirements of an electrical system. Electrical transformers are an important part of power generation, transmission, distribution, and energy systems.

The purpose of an electrical transformer is therefore related to the conversion and transfer of electrical energy. Its operating principle is fundamentally different from the mechanisms used in a Deep Learning Transformer.

### Transformer Architecture — Artificial Intelligence

In Artificial Intelligence, the term Transformer refers to a Deep Learning architecture introduced in the research paper **"Attention Is All You Need"** by Vaswani et al. The architecture was originally developed for sequence-to-sequence tasks and introduced an approach centered around attention mechanisms.

The Transformer architecture allows a model to process relationships between different elements of a sequence. This became particularly important for Natural Language Processing, where relationships between words or tokens can occur over different distances within a sequence.

---

## What Makes the AI Transformer Different?

Traditional sequence models such as Recurrent Neural Networks process information sequentially. Transformers introduced a different approach by using attention mechanisms to determine how different elements of a sequence relate to one another.

The central idea is that a token does not necessarily need to rely only on the immediately preceding token. Instead, an attention mechanism can examine relationships between different positions in the sequence and assign different levels of importance to them.

This idea is one of the foundations of modern Transformer-based Deep Learning models.

A simplified representation of the attention operation is:

$$
Attention(Q,K,V)
=
Softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

Here, **Q** represents the Query, **K** represents the Key, and **V** represents the Value. The term \(d_k\) represents the dimensionality of the Key vectors. This equation will be explored in greater detail later in the series because understanding the operation of Query, Key, Value, scaling, and Softmax is essential for understanding Self-Attention.

---
