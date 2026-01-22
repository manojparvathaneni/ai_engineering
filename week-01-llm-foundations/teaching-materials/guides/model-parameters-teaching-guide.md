# Understanding Model Parameters & Size - Teaching Guide

## The Big Question

When someone says "Llama 3 70B" or "GPT-4 has ~1.8 trillion parameters," what does that actually mean?

**Parameters = the learned numbers that make the model work**

Every weight and bias in every layer is a parameter. When we train a model, we're finding good values for all these numbers.

---

## Part 1: What Are Parameters?

### Parameters = Weights + Biases

Remember our fundamental equation:
```
y = Wx + b
```

- **W** (weights): The matrix that transforms inputs
- **b** (biases): The offset values added after

Both W and b are **parameters** - numbers learned during training.

### Counting Parameters in a Linear Layer

```
Linear Layer: input_size → output_size

Parameters = (input_size × output_size) + output_size
             \_____weights_____/   \__biases__/
```

**Example:** Linear layer from 768 → 3072
```
Weights: 768 × 3072 = 2,359,296
Biases:  3072
Total:   2,362,368 parameters
```

That's **2.36 million parameters** in just ONE layer!

---

## Part 2: Where Do Parameters Live in a Transformer?

A decoder-only transformer has parameters in several places:

```
┌─────────────────────────────────────────────────────────┐
│                    TOKEN EMBEDDING                       │
│              vocab_size × d_model parameters            │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                  TRANSFORMER BLOCK (×N)                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │            SELF-ATTENTION                         │  │
│  │   Q projection: d_model × d_model                 │  │
│  │   K projection: d_model × d_model                 │  │
│  │   V projection: d_model × d_model                 │  │
│  │   Output proj:  d_model × d_model                 │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │            FEED-FORWARD (MLP)                     │  │
│  │   Up projection:   d_model × d_ff                 │  │
│  │   Down projection: d_ff × d_model                 │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │            LAYER NORMS                            │  │
│  │   2 × d_model (scale and shift params)            │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   OUTPUT (LM HEAD)                       │
│              d_model × vocab_size parameters            │
│         (often tied/shared with embedding)              │
└─────────────────────────────────────────────────────────┘
```

### Key Dimensions to Know

| Symbol | Name | What It Is |
|--------|------|------------|
| **d_model** | Hidden dimension | Size of token vectors throughout the model |
| **d_ff** | Feed-forward dimension | Size of MLP hidden layer (usually 4 × d_model) |
| **n_layers** | Number of layers | How many transformer blocks stacked |
| **n_heads** | Attention heads | Parallel attention computations |
| **vocab_size** | Vocabulary size | Number of tokens the model knows |
| **d_head** | Head dimension | d_model / n_heads |

---

## Part 3: Calculating Total Parameters

### Formula for a Transformer

```
Total ≈ Embedding + (N × Block) + Output

Where Block = Attention + MLP + LayerNorms
```

### Detailed Breakdown

**1. Token Embedding**
```
vocab_size × d_model
```

**2. Each Transformer Block**
```
Attention:
  Q, K, V projections: 3 × (d_model × d_model + d_model)
  Output projection:   d_model × d_model + d_model
  
MLP (Feed-Forward):
  Up projection:   d_model × d_ff + d_ff
  Down projection: d_ff × d_model + d_model
  
Layer Norms:
  2 × (d_model + d_model)  [scale and shift for each]
```

**3. Output Layer (LM Head)**
```
d_model × vocab_size  (often tied with embedding, so may not add new params)
```

### Simplified Approximation

For most transformers:
```
Parameters ≈ 12 × n_layers × d_model²
```

This works because:
- Attention has ~4 × d_model² params (Q, K, V, Output)
- MLP has ~8 × d_model² params (up and down projections with d_ff = 4 × d_model)
- Total per block ≈ 12 × d_model²

---

## Part 4: Real Model Examples

### GPT-2 Family

| Model | d_model | n_layers | n_heads | Parameters |
|-------|---------|----------|---------|------------|
| GPT-2 Small | 768 | 12 | 12 | 117M |
| GPT-2 Medium | 1024 | 24 | 16 | 345M |
| GPT-2 Large | 1280 | 36 | 20 | 762M |
| GPT-2 XL | 1600 | 48 | 25 | 1.5B |

**Let's verify GPT-2 Small (117M):**
```
d_model = 768
n_layers = 12
vocab_size = 50,257

Embedding: 50,257 × 768 = 38.6M
Per block: ~12 × 768² = ~7.1M
12 blocks: 12 × 7.1M = 85.2M
Total: ~124M (close to 117M, difference is details)
```

### GPT-3 Family

| Model | d_model | n_layers | n_heads | Parameters |
|-------|---------|----------|---------|------------|
| GPT-3 Small | 768 | 12 | 12 | 125M |
| GPT-3 Medium | 1024 | 24 | 16 | 350M |
| GPT-3 Large | 1536 | 24 | 16 | 760M |
| GPT-3 XL | 2048 | 24 | 24 | 1.3B |
| GPT-3 2.7B | 2560 | 32 | 32 | 2.7B |
| GPT-3 6.7B | 4096 | 32 | 32 | 6.7B |
| GPT-3 13B | 5140 | 40 | 40 | 13B |
| **GPT-3 175B** | **12288** | **96** | **96** | **175B** |

**Let's verify GPT-3 175B:**
```
d_model = 12,288
n_layers = 96

Approximation: 12 × 96 × 12,288² = 173.9B ✓
```

### Llama Family (Meta)

| Model | d_model | n_layers | n_heads | Parameters |
|-------|---------|----------|---------|------------|
| Llama 2 7B | 4096 | 32 | 32 | 6.7B |
| Llama 2 13B | 5120 | 40 | 40 | 13B |
| Llama 2 70B | 8192 | 80 | 64 | 70B |
| Llama 3 8B | 4096 | 32 | 32 | 8B |
| Llama 3 70B | 8192 | 80 | 64 | 70B |
| Llama 3 405B | 16384 | 126 | 128 | 405B |

### What the Numbers Tell You

**More layers** = deeper reasoning, better at complex tasks
**Larger d_model** = richer representations, more nuance
**More heads** = more parallel attention patterns

Generally, scaling all three together gives best results.

---

## Part 5: Parameter Count vs. Model Size (Memory)

### Parameters → Bytes

Each parameter is typically stored as a floating-point number:

| Precision | Bytes per Parameter | Use Case |
|-----------|-------------------|----------|
| FP32 (float32) | 4 bytes | Training (full precision) |
| FP16 (float16) | 2 bytes | Training (mixed precision) |
| BF16 (bfloat16) | 2 bytes | Training (better range than FP16) |
| INT8 | 1 byte | Inference (quantized) |
| INT4 | 0.5 bytes | Inference (heavily quantized) |

### Memory Calculation

```
Memory = Parameters × Bytes per Parameter
```

**Example: Llama 2 70B**
```
FP32: 70B × 4 = 280 GB  (needs multiple high-end GPUs)
FP16: 70B × 2 = 140 GB  (still needs multiple GPUs)
INT8: 70B × 1 = 70 GB   (fits on 1-2 high-end GPUs)
INT4: 70B × 0.5 = 35 GB (can run on consumer GPU!)
```

This is why **quantization** matters for running models locally.

### Training vs. Inference Memory

Training requires MUCH more memory:
- Model parameters
- Gradients (same size as parameters)
- Optimizer states (2-8× parameters for Adam)
- Activations (for backpropagation)

**Rule of thumb:** Training needs 4-20× more memory than inference.

---

## Part 6: Reading Model Cards & Specs

When you see a model card, here's what to look for:

### Example: Llama 3 70B Model Card

```yaml
Model: Llama-3-70B
Architecture: Decoder-only Transformer
Parameters: 70B
Hidden size (d_model): 8192
Intermediate size (d_ff): 28672
Number of layers: 80
Number of attention heads: 64
Number of KV heads: 8  # Grouped-Query Attention
Vocab size: 128,256
Context length: 8192 tokens
Training tokens: 15T
```

**What each tells you:**

| Spec | Implication |
|------|-------------|
| 70B parameters | Large model, needs significant compute |
| d_model = 8192 | Rich internal representations |
| 80 layers | Very deep, good at complex reasoning |
| 64 heads | Many parallel attention patterns |
| 8 KV heads | Uses Grouped-Query Attention (efficiency) |
| 128K vocab | Large vocabulary (good for multilingual) |
| 8K context | Can process ~6000 words at once |
| 15T training tokens | Massive training data |

### Grouped-Query Attention (GQA)

Modern models use GQA to reduce memory:
- Standard: Every head has its own K, V projections
- GQA: Multiple heads share K, V projections
- MQA: All heads share one K, V (most aggressive)

```
n_kv_heads < n_heads → Using GQA
```

Llama 3 70B: 64 query heads, 8 KV heads = 8× memory savings in KV cache

---

## Part 7: Scaling Laws

Researchers found predictable relationships:

### The Chinchilla Scaling Law

**Optimal training:** tokens ≈ 20 × parameters

| Parameters | Optimal Training Tokens |
|------------|------------------------|
| 1B | 20B tokens |
| 7B | 140B tokens |
| 70B | 1.4T tokens |
| 175B | 3.5T tokens |

Many models are now "overtrained" (more tokens than optimal) because:
- Inference cost depends on model size
- Training is one-time cost
- Smaller overtrained model can match larger undertrained model

### Parameter-Performance Relationship

Generally follows:
```
Performance ∝ log(Parameters) × log(Training Tokens) × log(Compute)
```

Doubling parameters gives diminishing returns - you need ~10× parameters for noticeable improvement.

---

## Part 8: The Decoder-Only Flow Revisited

Now with parameter awareness:

```
"Hello world"
     ↓
┌─────────────────────────────────────┐
│          TOKENIZER                   │
│   (no parameters - just rules)       │
└─────────────────────────────────────┘
     ↓
[15496, 995]  (token IDs)
     ↓
┌─────────────────────────────────────┐
│      EMBEDDING LAYER                 │
│   50,257 × 768 = 38.6M params       │
│   Each token ID → 768-dim vector    │
└─────────────────────────────────────┘
     ↓
[[0.23, -0.45, ...], [0.67, 0.12, ...]]  (2 × 768)
     ↓
┌─────────────────────────────────────┐
│    TRANSFORMER BLOCKS (×12)         │
│   ~7M params each = 85M total       │
│   Tokens attend to each other       │
│   Build contextual representations  │
└─────────────────────────────────────┘
     ↓
[[0.89, -0.23, ...], [0.45, 0.78, ...]]  (2 × 768, transformed)
     ↓
┌─────────────────────────────────────┐
│         LM HEAD                      │
│   768 × 50,257 = 38.6M params       │
│   (often shared with embedding)     │
│   Project to vocabulary size        │
└─────────────────────────────────────┘
     ↓
[0.001, 0.002, ..., 0.15, ...]  (50,257 probabilities)
     ↓
┌─────────────────────────────────────┐
│         SOFTMAX                      │
│   (no parameters - just math)       │
│   Convert to probability dist       │
└─────────────────────────────────────┘
     ↓
Next token: "!" (highest probability)
```

---

## Part 9: Practical Implications

### Choosing a Model Size

| Use Case | Suggested Size | Why |
|----------|---------------|-----|
| Learning/Experimentation | 1-7B | Runs on consumer hardware |
| Personal assistant | 7-13B | Good balance of quality/speed |
| Complex reasoning | 70B+ | Needed for difficult tasks |
| Production (cost-sensitive) | 7-13B | Cheaper to run at scale |
| Production (quality-critical) | 70B+ | Best results |

### Hardware Requirements (Inference)

| Model Size | Minimum GPU (Quantized) | Recommended GPU |
|------------|------------------------|-----------------|
| 7B | 8GB (INT4) | 16GB (FP16) |
| 13B | 16GB (INT4) | 32GB (FP16) |
| 70B | 48GB (INT4) | 140GB (FP16) |
| 405B | 200GB+ (INT4) | 800GB+ (FP16) |

### Cost at Scale

Rough inference costs (cloud, per 1M tokens):
- 7B model: ~$0.10-0.20
- 70B model: ~$1-2
- 405B model: ~$5-15

This is why companies offer tiered models (Claude Haiku vs Sonnet vs Opus).

---

## Key Takeaways

1. **Parameters = weights + biases** that the model learns

2. **Most parameters live in:**
   - Embedding layer (vocab × d_model)
   - Attention projections (4 × d_model²)
   - MLP layers (8 × d_model²)

3. **Quick estimate:** Parameters ≈ 12 × layers × d_model²

4. **Size naming:** "70B" means 70 billion parameters

5. **Memory:** Params × precision (4B for FP32, 2B for FP16, etc.)

6. **Scaling:** Performance improves with log(parameters)

7. **Trade-offs:** Bigger = smarter but slower and more expensive

---

## Discussion Questions

1. Why do you think d_ff is typically 4× d_model? What would change if it were 2× or 8×?

2. If you had to design a model for a mobile phone (max 4GB RAM), what parameter count could you fit? What trade-offs would you make?

3. Why might a company train a smaller model on more tokens rather than a larger model on fewer tokens?

4. What does "Grouped-Query Attention" save memory on? Why is this important for long context?

---

## Exercises

1. **Calculate parameters:** A model has d_model=2048, n_layers=24, vocab_size=32000. Estimate total parameters.

2. **Memory planning:** You have a 24GB GPU. What's the largest model you can run at INT8? At INT4?

3. **Read a model card:** Find the Mistral 7B model card and identify: d_model, n_layers, n_heads, context length.
