# Attention Mechanism - Teaching Guide

## The One-Sentence Explanation

**Attention lets each word "look at" every other word and decide which ones matter for understanding it.**

---

## Part 1: The Problem Attention Solves

### Why Do We Need Attention?

Consider this sentence:
```
"The cat sat on the mat because it was tired"
```

What does "it" refer to? **The cat**, not the mat.

Without attention, each word is processed independently. The model would have no way to connect "it" back to "cat".

With attention, when processing "it", the model can:
1. Look at all previous words
2. Decide "cat" is highly relevant
3. Pull information from "cat" into its understanding of "it"

### Before Attention: The RNN Problem

RNNs processed words sequentially, passing a "hidden state" forward:

```
The → [state] → cat → [state] → sat → [state] → ... → it → [state]
```

Problems:
- Information from early words gets diluted
- Long-range dependencies are hard to learn
- Can't parallelize (must process sequentially)

### The Attention Solution

Let every word directly access every other word:

```
        The   cat   sat   on    mat   because   it
it  →   0.05  0.70  0.10  0.02  0.03   0.05    0.05
        └─────────────────────────────────────────┘
                    attention weights
```

"it" directly attends to "cat" with weight 0.70 - no information loss!

---

## Part 2: The Query, Key, Value Intuition

### The Library Analogy

Imagine you're in a library looking for information:

| Concept | Analogy | In Attention |
|---------|---------|--------------|
| **Query (Q)** | Your question: "Books about cats?" | What this word is looking for |
| **Key (K)** | Book spine labels: "Animals", "Fiction" | What each word advertises about itself |
| **Value (V)** | Actual book contents | The actual information each word carries |

**The process:**
1. You (Q) scan all book labels (K)
2. Find matches (Q·K similarity)
3. Read the matching books' contents (V)
4. Synthesize what you learned

### Every Word Gets All Three

For the sentence "The cat sat":

```
"cat" creates:
  Q_cat: "I'm looking for subjects, actions related to me"
  K_cat: "I'm a noun, animal, subject of sentence"
  V_cat: [actual vector representation of 'cat']
```

The Q, K, V are created by multiplying the word embedding by learned weight matrices:

```
Q = embedding × Wq
K = embedding × Wk
V = embedding × Wv
```

These Wq, Wk, Wv matrices are the **learned parameters** of the attention layer.

---

## Part 3: The Attention Calculation

### Step-by-Step Walkthrough

**Input:** 3 words, each a 4-dimensional embedding
```
"The" → [0.2, 0.5, 0.1, 0.8]
"cat" → [0.9, 0.1, 0.4, 0.3]
"sat" → [0.3, 0.7, 0.2, 0.6]

Stacked as matrix X (3×4):
X = [[0.2, 0.5, 0.1, 0.8],
     [0.9, 0.1, 0.4, 0.3],
     [0.3, 0.7, 0.2, 0.6]]
```

**Step 1: Project to Q, K, V**
```
Q = X × Wq   (3×4) × (4×4) = (3×4)
K = X × Wk   (3×4) × (4×4) = (3×4)
V = X × Wv   (3×4) × (4×4) = (3×4)
```

Each word now has a Q, K, and V vector.

**Step 2: Compute attention scores**
```
Scores = Q × Kᵀ   (3×4) × (4×3) = (3×3)

Result: 3×3 matrix where Scores[i,j] = 
        "how much should word i attend to word j?"

        To:  The   cat   sat
From:      ┌─────────────────┐
The        │ 1.2   0.8   0.5 │
cat        │ 0.9   2.1   1.3 │
sat        │ 0.6   1.4   1.8 │
           └─────────────────┘
```

**Step 3: Scale by √d**
```
Scaled = Scores / √4 = Scores / 2

        To:  The   cat   sat
From:      ┌─────────────────┐
The        │ 0.6   0.4   0.25│
cat        │ 0.45  1.05  0.65│
sat        │ 0.3   0.7   0.9 │
           └─────────────────┘
```

Why scale? Without it, large dimensions cause extreme softmax outputs.

**Step 4: Apply softmax (per row)**
```
Weights = softmax(Scaled, dim=-1)

Each row now sums to 1.0:

        To:  The   cat   sat
From:      ┌─────────────────┐
The        │ 0.40  0.33  0.27│  = 1.0
cat        │ 0.25  0.45  0.30│  = 1.0
sat        │ 0.22  0.33  0.45│  = 1.0
           └─────────────────┘
```

Now we have attention **weights** - probabilities of attending to each word.

**Step 5: Weighted sum of Values**
```
Output = Weights × V   (3×3) × (3×4) = (3×4)

Output[i] = weighted combination of all V vectors,
            weighted by how much word i attends to each word
```

**Result:** Each word's output vector now contains information from words it attended to!

### The Complete Formula

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d_k) × V
```

Where d_k is the dimension of the key vectors.

---

## Part 4: Multi-Head Attention

### Why Multiple Heads?

One attention pattern isn't enough. Consider:

```
"The animal didn't cross the street because it was too wide"
```

What does "it" refer to? What about "wide"?
- "it" → probably "animal" (but could be "street"!)
- "wide" → "street" (streets are wide, not animals)

Different relationships need different attention patterns.

### How Multi-Head Works

Instead of one big attention, run multiple smaller attentions in parallel:

```
Input (d_model=512)
    │
    ├──→ Head 1 (d_k=64) ──→ Output 1
    ├──→ Head 2 (d_k=64) ──→ Output 2
    ├──→ Head 3 (d_k=64) ──→ Output 3
    ...
    └──→ Head 8 (d_k=64) ──→ Output 8
                              │
                    Concatenate (512)
                              │
                    Linear projection
                              │
                         Final Output
```

### What Each Head Learns

Researchers have found heads specialize:
- Some track syntactic dependencies (subject-verb)
- Some track semantic relationships (adjective-noun)
- Some track positional patterns (next word, previous word)
- Some are redundant (attention patterns show this)

### The Math

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

where head_i = Attention(Q × W_Q^i, K × W_K^i, V × W_V^i)
```

Each head has its own Wq, Wk, Wv projection matrices.

---

## Part 5: Causal Masking (Decoder-Only)

### The Problem

When generating text, the model predicts one word at a time:

```
Input:  "The cat sat on the"
Output: "mat" (predicted next word)
```

**Critical rule:** When predicting a word, you can only see previous words!

If the model could see future words during training, it would just copy them instead of learning to predict.

### The Solution: Causal Mask

Before softmax, set future positions to -infinity:

```
Scores before mask:
        The   cat   sat   on    the
The     1.2   0.8   0.5   0.3   0.2
cat     0.9   2.1   1.3   0.7   0.4
sat     0.6   1.4   1.8   0.9   0.5
on      0.4   0.8   1.1   1.5   0.8
the     0.3   0.7   0.9   1.2   1.6

After causal mask (lower triangle only):
        The   cat   sat   on    the
The     1.2   -∞    -∞    -∞    -∞
cat     0.9   2.1   -∞    -∞    -∞
sat     0.6   1.4   1.8   -∞    -∞
on      0.4   0.8   1.1   1.5   -∞
the     0.3   0.7   0.9   1.2   1.6
```

After softmax, -∞ becomes 0:
```
        The   cat   sat   on    the
The     1.0   0     0     0     0     (only sees itself)
cat     0.3   0.7   0     0     0     (sees The, cat)
sat     0.2   0.35  0.45  0     0     (sees The, cat, sat)
...
```

### Encoder vs Decoder Attention

| Type | Masking | Use Case |
|------|---------|----------|
| Encoder (BERT) | None - sees all words | Understanding, classification |
| Decoder (GPT) | Causal - only past | Generation, completion |
| Cross-attention | Decoder attends to encoder | Translation, summarization |

---

## Part 6: Positional Encoding

### The Problem

Attention treats the input as a **set**, not a sequence. It has no inherent sense of word order.

```
"Dog bites man" vs "Man bites dog"
```

Same words, different meanings! The model needs position information.

### The Solution: Add Position Information

Before attention, add a position-dependent vector to each embedding:

```
Final_embedding = Word_embedding + Position_embedding
```

### Sinusoidal Position Encoding (Original Transformer)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Why sine/cosine?
- Each position gets a unique pattern
- The model can learn relative positions (PE[pos+k] is a function of PE[pos])
- Generalizes to sequences longer than training data

### Learned Position Embeddings (GPT-style)

Just learn a separate embedding for each position:

```
Position 0 → [learned vector]
Position 1 → [learned vector]
...
Position 2047 → [learned vector]
```

Simpler, works well, but limited to max training length.

### Rotary Position Embeddings (RoPE) - Modern

Used in Llama, applies rotation to Q and K based on position. Allows better length generalization.

---

## Part 7: The Complete Self-Attention Layer

Putting it all together:

```
┌─────────────────────────────────────────────────────────────┐
│                    SELF-ATTENTION LAYER                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: X (seq_len × d_model)                               │
│           │                                                 │
│           ├──→ Q = X × Wq                                   │
│           ├──→ K = X × Wk                                   │
│           └──→ V = X × Wv                                   │
│                │                                            │
│                ↓                                            │
│  Split into heads (if multi-head)                           │
│                │                                            │
│                ↓                                            │
│  Scores = Q × Kᵀ / √d_k                                     │
│                │                                            │
│                ↓                                            │
│  Apply causal mask (if decoder)                             │
│                │                                            │
│                ↓                                            │
│  Weights = softmax(Scores)                                  │
│                │                                            │
│                ↓                                            │
│  Attended = Weights × V                                     │
│                │                                            │
│                ↓                                            │
│  Concatenate heads (if multi-head)                          │
│                │                                            │
│                ↓                                            │
│  Output = Attended × Wo                                     │
│                │                                            │
│                ↓                                            │
│  Output: (seq_len × d_model)  ← Same shape as input!        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 8: Connection to What We've Learned

### Parameters in Attention

For single-head attention with d_model dimensions:
```
Wq: d_model × d_model
Wk: d_model × d_model  
Wv: d_model × d_model
Wo: d_model × d_model

Total: 4 × d_model² parameters
```

For multi-head (this is equivalent, just organized differently):
```
Each head has smaller Wq, Wk, Wv (d_model × d_head)
But we have n_heads of them
Plus one Wo at the end

Still totals: ~4 × d_model² parameters
```

### The Flow in an LLM

```
"Hello world" 
    → Tokenize → [15496, 995]
    → Embed → [[0.1, 0.2, ...], [0.3, 0.4, ...]]
    → + Position encoding
    → Self-Attention (×N layers) ← WE ARE HERE
    → MLP
    → Layer Norm
    → ... repeat N times ...
    → Final layer → vocabulary probabilities
```

---

## Key Takeaways

1. **Attention = looking at other words** to understand context

2. **Q, K, V**:
   - Q: What am I looking for?
   - K: What do I offer?
   - V: What information do I have?

3. **The formula**: `softmax(QKᵀ/√d) × V`

4. **Multi-head**: Multiple attention patterns in parallel

5. **Causal mask**: Can only see past words (for generation)

6. **Position encoding**: Gives the model sense of word order

7. **Output shape = input shape**: Attention enriches, doesn't change dimensions

---

## Discussion Questions

1. Why do you think self-attention (attending to your own sequence) works better than cross-attention for language modeling?

2. If you removed the scaling factor (√d), what would happen to the attention weights? Why?

3. Why might a model learn to have some attention heads that barely do anything (near-uniform attention)?

4. How does the number of attention heads relate to the model's ability to capture different types of relationships?

---

## Exercises

1. **Calculate by hand**: Given Q=[1,2], K=[[1,0],[0,1],[1,1]], compute the attention scores and weights (assume no scaling).

2. **Draw the mask**: For a sequence of 5 tokens, draw the causal attention mask matrix.

3. **Parameter count**: A model has d_model=1024 and 16 attention heads. How many parameters are in one multi-head attention layer?
