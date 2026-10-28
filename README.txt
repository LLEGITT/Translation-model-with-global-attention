# APM 53674 - ALTeGraD : Lab Session 1
## Neural Machine Translation and Language Modeling

This repository contains the implementation and report for **Lab 1** of the ALTeGraD course (Prof. Michalis Vazirgiannis, TA: Yang Zhang & Amr Mohamed).

---

## 📌 Lab Objectives

The main objective of this practical session is to build and train a **Sequence-to-Sequence (Seq2Seq)** architecture with **Global Attention** (following [Luong et al., 2015](https://arxiv.org/abs/1508.04025)) for English-to-French machine translation.

### Key Concepts Covered:
- Implementation in **PyTorch** of an Encoder and a Decoder based on **GRU** (Gated Recurrent Unit) cells.
- **Global Attention** mechanism (score calculation via concatenation).
- **Greedy Decoding** strategy.
- Analysis of the limitations of classical Seq2Seq models (hallucinations, over-translation, under-translation).
- Visualization and interpretation of alignment/attention matrices.

---

## 🛠️ Notebook Structure & Tasks

1. **Task 1: Encoder** — Implementation of the forward pass (`forward`) for the `Encoder` class (nn.Embedding + nn.GRU).
2. **Task 2: Decoder** — Step-by-step implementation of the decoder conditioned on the source context and the previous word.
3. **Task 3: Attention Mechanism** — Implementation of global attention weight calculations and the context vector $c_t$.
4. **Task 4: Seq2Seq Model** — Global assembly of the architecture (`seq2seqModel`), including inference mode management (`predict`) and the training loop (`fit`).
5. **Task 5: Training** — Verification of loss convergence on the Tatoeba dataset.
6. **Task 6: Inference & Evaluation** — Evaluation of the pre-trained model on test sentences.

---

## ❓ Theoretical Questions & Answers

The notebook contains detailed answers to the following questions:
- **Question 1:** Critical analysis of **Greedy Decoding** vs. **Beam Search**.
- **Question 2:** Analysis of over-translation and under-translation issues in basic attention models.
- **Question 3:** Code and heatmap matrix to visualize attention alignments (adjective/noun inversion between French and English).
- **Question 4:** Handling polysemy (e.g., "**mean**") and phenomena like hallucinations or infinite loops.

---

## 📂 Data & Prerequisites

- **Dataset:** ~136,521 training pairs / ~34,130 test pairs (Tatoeba / Anki).
- **Main Dependencies:**
  ```bash
  python >= 3.6
  torch
  nltk
  numpy
  matplotlib
  seaborn
  tqdm