# Training an NLP Model from Scratch

This guide walks through the conceptual and technical journey of building a word embedding model (like Word2Vec) from the ground up.

---

## Table of Contents

1. [Gathering Data](#1-gathering-data)
2. [Cleaning Data](#2-cleaning-data)
3. [Tokenizing the Data](#3-tokenizing-the-data-words---integers)
4. [Training the Model (CBOW & Embeddings)](#4-training-the-model-cbow--embeddings)
5. [Technical Deep Dive](#technical-deep-dive)
6. [Advanced Concepts & Modern Techniques](#advanced-concepts--modern-techniques)

---

## 1. Gathering Data

The data we provide determines the model's bias. To create a well-rounded "understanding" of language, we need clean text from multiple sources.

> **Analogy:** Think of it like teaching a child a language. If they only read 18th-century law books, they'll speak formally but won't know what a "smartphone" is. If they only read Reddit, they'll know slang but struggle with coherent, polite sentences.

### Core Data Sources:

- **Wiki Data:** Grounds the model in "truths". It is formal, dry, and grammatically dense, though it lacks emotional intuition.
- **Novels:** Provides emotional context (e.g., *"The gloomy sky"*). It helps the model understand non-objective relationships between words.
- **News/Online Chatter:** Keeps the model updated with modern words, slang, and current events.

**Key Concept:** `Garbage In, Garbage Out`. The training corpus literally **shapes** the model's reality. For this project, we sample from raw text files in `data/raw_parts`.

---

## 2. Cleaning Data

Raw text is messy. Before training, we must clean the corpus to obtain consistent results.

We employ a **Standard Aggressive Filter**:

1. Convert everything to **lowercase**.
2. Remove all characters except **a-z** and **spaces**.

### Example:

- **Input:** `"Hello, World! It's 2026."`
- **Output:** `"hello world its "`

**Why so aggressive?**
It sounds destructive because we lose nuance (capitalization, punctuation, numbers). However, it results in **massive simplification**. Our vocabulary shrinks from millions of unique tokens (like *"Run"*, *"Run!"*, *"run"*) down to just one: *"run"*. This allows the model to focus purely on semantic relationships.

---

## 3. Tokenizing the Data (Words -> Integers)

Computers don't process strings; they process numbers. Before "vectorization," we must translate words into **Integers**.

> **The Phonebook Analogy:** Imagine a giant phonebook for English.
> 
> - `"a"` is on page 1.
> - `"apple"` is on page 42.
> - `"zebra"` is on page 90,000.

We build a **Vocabulary** (a dictionary mapping words to unique IDs). We then scan the entire corpus and replace every word with its ID.

**Efficiency:**
In our implementation, we store these as `uint32` arrays using `numpy`. This allows us to hold **50 million words** in RAM with almost zero overhead, ready for the CPU to process at lightning speeds.

---

## 4. Training the Model (CBOW & Embeddings)

This is where the magic happens. We don't "calculate" vectors; we **learn** them through an iterative process.

### The Architecture: CBOW

We use **Continuous Bag of Words**.

1. Take a sentence: *"The king owns these lands."*
2. Hide the target: *"The [blank] owns these lands."*
3. **The Question:** Based on the neighbors ("The", "owns", "these", "lands"), what word belongs in the center?

### The Embeddings

The model maintains a "Vector Meaning Map" (a matrix). Initially, every word is placed [Randomly](#initialization-random-chaos) in 300-dimensional space. "King" might start next to "toaster".

### The Learning Process

1. **The Guess:** The model predicts a word based on current vectors.
2. **The Loss:** We calculate the error. [See Loss Calculation](#calculating-loss-negative-sampling).
3. **The Nudge:** We [nudge](#optimization-nudging-the-neurons) the vector for "king" closer to its neighbors and push incorrect guesses (like "bicycle") further away.

### Negative Sampling

Comparing "king" against 100,000 other words is too slow. Instead, we compare it against the correct word and **15 random "noise" words** (e.g., "taco", "purple").

- **Pull** "king" closer to the context.
- **Push** "taco" further away.

---

## Technical Deep Dive

Precise explanations for the math and code under the hood.

### Vectorization (Embeddings)

Vectorization is implemented as a **Lookup Table** (Matrix).

- **Size:** `Vocab_Size (100,000) x Dimensions (300)`.
- **Retrieval:** `vector = Matrix[word_id]`.
- In PyTorch: `nn.Embedding(100000, 300)`.

### Initialization (Random Chaos)

We use **Xavier Uniform Initialization** to fill the matrix with small random floats (e.g., -0.05 to +0.05). This ensures gradients can flow immediately during the first pass.

### Calculating Loss (Negative Sampling)

**Goal:** Maximize similarity with the target and minimize similarity with noise.

1. **Dot Product:** `Score = dot(context_vector, target_vector)`.
2. **Activation:** Apply `LogSigmoid` to squash scores.
3. **Formula:**
  `Loss = - (LogSigmoid(Score_Pos) + Sum(LogSigmoid(-Score_Neg)))`

### Optimization (Nudging the Neurons)

We use **Backpropagation** and an optimizer called `SparseAdam`.

- **The Gradient:** "How much did word X contribute to the error?"
- **The Update:** `New_Vector = Old_Vector - (Learning_Rate * Gradient)`.

### The Training Loop (Step-by-Step)

1. **Batching:** Grab 10k-100k sentences.
2. **Context Creation:** Extract target words and their neighbors.
3. **Negative Sampling:** Select 15 random words per target.
4. **Forward Pass:** Calculate Dot Products and Scores.
5. **Backward Pass:** Calculate gradients (blame assignment).
6. **Step:** Update the specific rows in the Embedding matrix.

---

## Advanced Concepts & Modern Techniques

Word2Vec is the foundation, but the field has evolved significantly.

### N-Grams

Treats word combinations as single tokens.

- **Bi-grams:** *"New York"*, *"ice cream"*.
- **Benefit:** Captures compound meanings that individual words miss.

### FastText (Sub-word Information)

Breaks words into character chunks (e.g., `"apple"` -> `"<ap"`, `"app"`, `"ppl"`).

- **Benefit:** Can understand unknown words (out-of-vocabulary) by looking at their pieces (prefixes/suffixes) and handles typos gracefully.

### Transformers (The Modern Era)

Unlike Word2Vec's **Static Embeddings** (where "bank" always has the same vector), Transformers like **BERT** and **GPT** use **Contextual Embeddings**.

- The vector for "bank" changes dynamically based on whether you're at a "river bank" or a "bank account" using a mechanism called **Attention**.