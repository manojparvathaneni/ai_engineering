# Introduction to LLMs: Teaching Guide

## The Big Idea

> **An LLM has never "seen" a single word in any language. It only sees numbers.**
> 
> — Inspired by Jay Alammar's illustrated guides

This is the most important mental model for understanding how LLMs work. Everything else builds on this.

---

## Architecture Diagram

See: `llm-intro-architecture.mermaid`

The diagram shows two worlds:
- **Human World**: What we see (text, images, audio, video)
- **Number World**: What the LLM sees (tokens, embeddings, vectors)

Encoders and decoders are the **translators** between these worlds.

---

## The Journey of Text Through an LLM

```
"Hello world"                    ← What you type
      ↓
   Tokenizer                     ← Splits into subwords
      ↓
["Hello", " world"]              ← Tokens (still readable)
      ↓
[15496, 995]                     ← Token IDs (just numbers now)
      ↓
[[0.02, -0.8, 0.1, ...],         ← Embeddings (dense vectors)
 [0.15, 0.33, -0.7, ...]]           Each token → ~thousands of dimensions
      ↓
    🧠 LLM                        ← Processes vectors, predicts next token
      ↓
[18435]                          ← Output token ID
      ↓
  Detokenizer                    ← Looks up the token
      ↓
"Hi"                             ← What you read
```

**Key point**: From token IDs onward, there are no words. The model operates entirely in numerical/vector space.

---

## Definitions

### What is an LLM?

**Simple definition**: An AI model that can understand and generate text.

**Technical definition**: A neural network trained on massive text data to predict the next token in a sequence. 

**Important nuance**: The "L" stands for "Language" — an LLM specifically handles text. Multimodal models (images, audio, video) combine an LLM with additional encoders/decoders.

### Core Components

| Term | What It Is | Analogy |
|------|------------|---------|
| **Tokenizer** | Splits text into chunks (tokens) and assigns each a number | A dictionary that converts words to index numbers |
| **Token** | The basic unit the LLM operates on — usually subwords | Like syllables, but determined by frequency in training data |
| **Token ID** | The numerical identifier for a token | The index number in the tokenizer's vocabulary |
| **Embedding** | A dense vector representing a token's "meaning" | Coordinates in a high-dimensional space where similar concepts are nearby |
| **LLM** | The neural network that processes embeddings and predicts the next token | The "reasoning core" — a massive pattern matcher |

### Encoders (Input Translators)

| Encoder | Input | Output | Examples |
|---------|-------|--------|----------|
| **Tokenizer** | Text | Token IDs | BPE, SentencePiece, tiktoken |
| **Vision Encoder** | Images | Embeddings | CLIP, ViT, SigLIP |
| **Audio Encoder** | Sound | Embeddings | Whisper, wav2vec |
| **Video Encoder** | Video | Embeddings | Sequences of frame embeddings + audio |

### Decoders (Output Translators)

| Decoder | Input | Output | Notes |
|---------|-------|--------|-------|
| **Detokenizer** | Token IDs | Text | Simple lookup — not a neural network |
| **Image Decoder** | Image tokens | Images | Often diffusion models |
| **Audio Decoder** | Audio tokens | Sound | Vocoders, neural audio synthesis |
| **Video Decoder** | Video tokens | Video | Frame generation models |

---

## Why This Matters

Understanding that LLMs only see numbers explains many behaviors:

### 1. Tokenization Quirks
- "How many r's in strawberry?" → LLM might fail because it sees `[straw, berry]` not individual letters
- Different languages tokenize differently (efficiency varies)

### 2. The "Understanding" Illusion
- The model doesn't "know" English or "understand" concepts
- It learned statistical patterns: "after these tokens, this token is likely"
- Apparent understanding emerges from massive scale pattern matching

### 3. Multimodal Design
- Vision/audio encoders do the same job as tokenizers: translate human-interpretable input into numbers
- The LLM core is modality-agnostic — it just processes vectors
- This is why you can add new modalities without retraining the whole model

### 4. Why Training Data Matters
- The model can only learn patterns present in its training data
- Token frequencies affect what it learns well vs. poorly

---

## Summary for Students

1. **LLM = Language Model** — specifically handles text (multimodal models add other capabilities)
2. **The model never sees words** — only numerical representations (tokens → embeddings)
3. **Encoders translate in**, decoders translate out — the LLM core operates in "number world"
4. **Understanding emerges from scale** — billions of parameters learning patterns from trillions of tokens

---

## Further Reading

- Jay Alammar's "The Illustrated Transformer" — visual explanations of attention and architecture
- Andrej Karpathy's "Let's build GPT" — hands-on coding walkthrough
- 3Blue1Brown's neural network series — mathematical intuition

---

*Last updated: Week 1 - LLM Foundations*
