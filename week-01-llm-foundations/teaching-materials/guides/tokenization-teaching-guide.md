# Tokenization for LLMs: Teaching Guide

## Overview

Tokenization is the bridge between human text and machine numbers. This is where the "LLMs have never seen a single word" concept becomes concrete.

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                             │
│                                                                  │
│   Crawling  ──►  Data Cleaning  ──►  Tokenization               │
│                                      (this module)              │
└─────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                                    Model Architecture & Training
```

---

## The Two Phases

Tokenization has two distinct phases:

| Phase | When | What Happens | Output |
|-------|------|--------------|--------|
| **Training Phase** | Before model training | Build vocabulary from corpus | Vocabulary + merge rules |
| **Inference Phase** | During model use | Apply vocabulary to new text | Token IDs ↔ Text |

```
TRAINING PHASE                           INFERENCE PHASE
──────────────                           ───────────────

Long text corpus                         New text input
      │                                        │
      ▼                                        ▼
 Text splitting                           Apply merge rules
      │                                        │
      ▼                                        ▼
 Build vocabulary                         Token IDs (encode)
      │                                        │
      ▼                                        ▼
 Merge rules                              Text (decode)
```

---

## Text Splitting Approaches

### The Three Options

| Approach | How It Works | Example |
|----------|--------------|---------|
| **Word-level** | Split on whitespace/punctuation | "Hello world" → ["Hello", "world"] |
| **Character-level** | Each character is a token | "Hello" → ["H", "e", "l", "l", "o"] |
| **Subword-level** | Learn common subword units | "transformer" → ["trans", "former"] |

### Comparison

```
┌─────────────┬────────────┬─────────────┬──────────────┬─────────────────┐
│ Approach    │ Vocab Size │ Seq Length  │ OOV Problem  │ Example Models  │
├─────────────┼────────────┼─────────────┼──────────────┼─────────────────┤
│ Word        │ 100K+      │ Short       │ Severe       │ Early word2vec  │
│ Character   │ ~256       │ Very Long   │ None         │ CharacterBERT   │
│ Subword     │ 32K-100K   │ Medium      │ Minimal      │ GPT, LLaMA, etc │
└─────────────┴────────────┴─────────────┴──────────────┴─────────────────┘
```

### Why Subword Won

**Word-level problems:**
```python
vocabulary = {"hello", "world", "transformer", ...}  # 100K+ words

# But what about:
"ChatGPT"        # ❌ Not in vocabulary → <UNK>
"transformers"   # ❌ Different from "transformer"
"transformrs"    # ❌ Typo → <UNK>
"café"           # ❌ Different languages
```

**Character-level problems:**
```python
"Hello, world!" → ['H','e','l','l','o',',',' ','w','o','r','l','d','!']
# 13 tokens for 2 words!
# Attention is O(n²) - very expensive
# "H" alone doesn't mean much
```

**Subword-level (the sweet spot):**
```python
"ChatGPT" → ["Chat", "GPT"]           # Known subwords
"transformers" → ["transform", "ers"]  # Suffix patterns
"transformrs" → ["transform", "rs"]    # Handles typos
"café" → ["caf", "é"]                  # Handles Unicode
```

---

## BPE (Byte Pair Encoding) Algorithm

BPE is the most common subword tokenization algorithm. Used by GPT, LLaMA, and most modern LLMs.

### The Core Idea

**Start with characters, repeatedly merge the most frequent pair.**

### Step-by-Step Example

**Training corpus:** ["low", "lower", "newest", "widest"]

**Step 1: Initialize with characters**

```
Vocabulary: {l, o, w, e, r, n, s, t, i, d, </w>}

Words as characters:
  "low"    → l o w </w>
  "lower"  → l o w e r </w>
  "newest" → n e w e s t </w>
  "widest" → w i d e s t </w>
```

**Step 2: Count adjacent pairs**

```
Pair counts:
  (l, o): 2    ← Most frequent!
  (o, w): 2
  (e, s): 2
  (s, t): 2
  ...
```

**Step 3: Merge most frequent pair**

```
Merge rule #1: l + o → lo

New vocabulary: {lo, w, e, r, n, s, t, i, d, </w>}

Words now:
  "low"    → lo w </w>
  "lower"  → lo w e r </w>
  "newest" → n e w e s t </w>
  "widest" → w i d e s t </w>
```

**Step 4: Repeat until desired vocabulary size**

```
Merge rule #2: lo + w → low
Merge rule #3: e + s → es
Merge rule #4: es + t → est
Merge rule #5: est + </w> → est</w>
...
```

**Final vocabulary:**

```
{d, i, n, w, e, r, low, est</w>, lower</w>, ...}

With merge rules:
  1. l + o → lo
  2. lo + w → low
  3. e + s → es
  4. es + t → est
  5. est + </w> → est</w>
  ...
```

### Using the Vocabulary (Inference)

To tokenize new text, apply merge rules in order:

```python
def tokenize(word, merge_rules):
    tokens = list(word) + ['</w>']  # Start with characters
    
    for (a, b) in merge_rules:
        # Find and merge all occurrences of (a, b)
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == a and tokens[i+1] == b:
                tokens = tokens[:i] + [a+b] + tokens[i+2:]
            else:
                i += 1
    
    return tokens

# Example:
tokenize("lowest", merge_rules)
# → ['l','o','w','e','s','t','</w>']
# → ['lo','w','e','s','t','</w>']      (rule 1)
# → ['low','e','s','t','</w>']         (rule 2)
# → ['low','es','t','</w>']            (rule 3)
# → ['low','est','</w>']               (rule 4)
# → ['low','est</w>']                  (rule 5)
# Final: ['low', 'est</w>']
```

---

## Using tiktoken

tiktoken is OpenAI's tokenizer library. It implements BPE and is used for GPT models.

### Installation

```bash
pip install tiktoken
```

### Basic Usage

```python
import tiktoken

# Get the encoder for GPT-4 / GPT-3.5-turbo
enc = tiktoken.get_encoding("cl100k_base")

# Vocabulary size
print(enc.n_vocab)  # 100,277 tokens
```

### Encoding (Text → Token IDs)

```python
text = "Hello, world!"
tokens = enc.encode(text)
print(tokens)  # [9906, 11, 1917, 0]
print(len(tokens))  # 4 tokens
```

### Decoding (Token IDs → Text)

```python
token_ids = [9906, 11, 1917, 0]
text = enc.decode(token_ids)
print(text)  # "Hello, world!"
```

### Inspecting Individual Tokens

```python
text = "ChatGPT"
tokens = enc.encode(text)

# See each token as a string
for token_id in tokens:
    token_str = enc.decode([token_id])
    print(f"  {token_id} → '{token_str}'")

# Output:
#   34 → 'Chat'
#   38 → 'G'
#   2898 → 'PT'
```

### Different Encoders

```python
encoders = {
    "cl100k_base": "GPT-4, GPT-3.5-turbo (100K vocab)",
    "p50k_base": "Codex, code-davinci-002 (50K vocab)",
    "r50k_base": "GPT-3 davinci (50K vocab)",
}

text = "def hello():"
for enc_name in encoders:
    encoder = tiktoken.get_encoding(enc_name)
    tokens = encoder.encode(text)
    print(f"{enc_name}: {len(tokens)} tokens - {tokens}")
```

---

## Practical Examples

### Example 1: Token Boundaries

```python
enc = tiktoken.get_encoding("cl100k_base")

# Words are often multiple tokens
examples = [
    "transformer",      # → ['trans', 'former']
    "transformers",     # → ['transform', 'ers']
    "ChatGPT",          # → ['Chat', 'G', 'PT']
    "antidisestablish", # → ['ant', 'id', 'ises', 'tab', 'lish']
]

for text in examples:
    tokens = enc.encode(text)
    token_strs = [enc.decode([t]) for t in tokens]
    print(f"'{text}' → {token_strs}")
```

### Example 2: Code Tokenization

```python
code = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""

tokens = enc.encode(code)
print(f"Code: {len(code)} characters, {len(tokens)} tokens")
# Ratio: ~3.5 chars/token for code
```

### Example 3: Multilingual

```python
texts = {
    "English": "Hello, how are you?",
    "Spanish": "Hola, ¿cómo estás?",
    "Japanese": "こんにちは、お元気ですか？",
    "Chinese": "你好，你好吗？",
}

for lang, text in texts.items():
    tokens = enc.encode(text)
    print(f"{lang}: {len(tokens)} tokens for {len(text)} chars")

# Output shows non-English needs more tokens (English-heavy training data)
```

### Example 4: Why "strawberry" Fails

```python
text = "strawberry"
tokens = enc.encode(text)
token_strs = [enc.decode([t]) for t in tokens]
print(token_strs)  # ['str', 'aw', 'berry']

# The model sees 3 tokens, not individual characters!
# It can't easily count 'r's because it doesn't "see" the r's directly
```

---

## Tokenization in the LLM Pipeline

```
User Input: "Hello, world!"
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                     TOKENIZER                                │
    │              "Hello, world!" → [9906, 11, 1917, 0]          │
    │                                                              │
    │   This is where text becomes numbers!                       │
    │   The LLM NEVER sees "Hello" - only token ID 9906           │
    └─────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   EMBEDDING LAYER                            │
    │         [9906, 11, 1917, 0] → [[0.02, -0.8, ...],           │
    │                                [0.15, 0.33, ...],           │
    │                                [0.77, -0.21, ...],          │
    │                                [-0.5, 0.91, ...]]           │
    │                                                              │
    │   Each token ID → dense vector (e.g., 4096 dimensions)      │
    └─────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   TRANSFORMER LAYERS                         │
    │                                                              │
    │   Self-attention, feed-forward networks, layer norm...      │
    │   This is where "understanding" emerges                     │
    └─────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   OUTPUT LAYER                               │
    │                                                              │
    │   Predict probability for EACH of 100,277 tokens            │
    │   Highest probability → next token ID                       │
    └─────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   DETOKENIZER                                │
    │              [1722] → "Hi"                                   │
    │                                                              │
    │   Simple lookup table - NOT a neural network                │
    └─────────────────────────────────────────────────────────────┘
```

---

## Key Concepts for Teaching

### 1. Tokenization Happens FIRST

- Same tokenizer for training and inference
- Vocabulary is fixed after tokenizer training
- Model can only output tokens in its vocabulary

### 2. The Model's "Vocabulary" = Tokenizer's Vocabulary

```
cl100k_base vocabulary: 100,277 tokens
                              ↓
       Model output layer: 100,277 possible outputs
```

### 3. Token Boundaries Affect Model Behavior

```
"ChatGPT" → ['Chat', 'G', 'PT']  (3 separate units)
"Hello"   → ['Hello']            (1 unit)
```

This is why LLMs can struggle with:
- Character counting ("How many r's in strawberry?")
- Spelling tasks
- Anagram solving

### 4. Different Models = Different Tokenizers

| Model | Tokenizer | Vocab Size |
|-------|-----------|------------|
| GPT-4 | cl100k_base | 100,277 |
| GPT-3 | r50k_base | 50,257 |
| LLaMA | SentencePiece | 32,000 |
| Claude | Custom BPE | ~100K |

**Must use matching tokenizer for each model!**

### 5. Rule of Thumb for Token Estimation

```
English text:
  ~1 token ≈ 4 characters
  ~1 token ≈ 0.75 words
  ~100 tokens ≈ 75 words

Code:
  ~1 token ≈ 3-4 characters
  More tokens than equivalent English
```

---

## Tools for Exploration

### 1. tiktokenizer.vercel.app

Visual web tool to explore tokenization:
- Paste text, see tokens highlighted
- Try different encodings
- Great for demonstrations

### 2. tiktoken Library

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("your text here")
```

### 3. HuggingFace Tokenizers

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
tokens = tokenizer.encode("your text here")
```

---

## Discussion Questions

1. **Why does subword tokenization handle typos better than word-level?**
   (Hint: "transformrs" → ["transform", "rs"])

2. **Why do non-English languages typically produce more tokens?**
   (Hint: What data was BPE trained on?)

3. **How does vocabulary size affect model training?**
   - Larger vocab = shorter sequences, but more parameters
   - Smaller vocab = longer sequences, but fewer parameters

4. **Why can't LLMs easily count letters in a word?**
   (Hint: Token boundaries)

5. **If you're building a domain-specific model (e.g., for code), would you:**
   - Use an existing tokenizer?
   - Train a new one on your domain data?
   - Why?

---

## Exercises

### Exercise 1: Manual BPE

Given corpus: ["cat", "cats", "rat", "rats"]

1. Initialize character vocabulary
2. Run 3 BPE merge iterations
3. What's your final vocabulary?
4. How would you tokenize "bats"?

### Exercise 2: Token Counting

Using tiktoken, count tokens for:
- A 100-word English paragraph
- The same content translated to Japanese
- A 50-line Python function

Calculate chars/token ratio for each.

### Exercise 3: Edge Cases

Tokenize and explain:
```python
enc.encode("aaaaaaaaaaaaaaaa")  # Repeated character
enc.encode("a a a a a a a a")  # Spaced characters
enc.encode("123456789")         # Numbers
enc.encode("   ")               # Whitespace
enc.encode("🚀🎉💻")            # Emojis
```

---

## Key Takeaways

1. **Two phases**: Training (build vocab) and Inference (use vocab)

2. **Subword is the standard**: Balances vocabulary size, sequence length, and OOV handling

3. **BPE algorithm**: Start with characters, merge most frequent pairs

4. **Token boundaries matter**: Affects what the model can "see"

5. **tiktoken**: The practical tool for working with OpenAI tokenizers

6. **The big insight**: LLMs never see words - only token IDs that get embedded into vectors

---

## Connection to Next Topic: Model Architecture

Tokenization produces sequences of token IDs. The next step is:

1. **Embedding layer**: Token IDs → Dense vectors
2. **Transformer layers**: Process embeddings with attention
3. **Output layer**: Predict next token probabilities

This is where we move from "data preparation" to "model architecture"!

---

## Resources

- **tiktoken library**: github.com/openai/tiktoken
- **tiktokenizer**: tiktokenizer.vercel.app
- **HuggingFace Tokenizers Course**: huggingface.co/learn/nlp-course/chapter6/1
- **Original BPE Paper**: "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2015)

---

*Next module: Model Architecture - How transformers process token embeddings*
