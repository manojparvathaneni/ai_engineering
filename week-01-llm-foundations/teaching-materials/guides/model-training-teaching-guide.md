# Model Training - Teaching Guide

## The One-Sentence Summary

**Training adjusts the model's weights so it becomes better at predicting the next token.**

---

## Part 1: Why Train? The Before and After

### Before Training: Random Chaos

A freshly initialized model has random weights. Ask it to predict the next word:

```
Input: "The president of the United States is"

Untrained model's predictions:
  "refrigerator" → 0.8%
  "purple"       → 0.7%
  "the"          → 0.6%
  "banana"       → 0.5%
  ... (all roughly equal, ~0.002% each for 50K tokens)
```

The model has no idea. Every word is equally likely because the weights are random.

### After Training: Learned Patterns

After seeing billions of sentences, the model learns patterns:

```
Input: "The president of the United States is"

Trained model's predictions:
  "the"      → 12%   (often followed by a title)
  "a"        → 8%
  "Joe"      → 5%    (learned from recent text)
  "Donald"   → 4%    (learned from text)
  "Barack"   → 2%
  ...
  "banana"   → 0.0001%
```

**Training transformed random weights into a next-token predictor.**

---

## Part 2: The Training Objective

### What We're Optimizing

For each position in the training text, we know the correct next token.

```
Training text: "The cat sat on the mat"

Position 1: Input="The"           → Correct next="cat"
Position 2: Input="The cat"       → Correct next="sat"
Position 3: Input="The cat sat"   → Correct next="on"
Position 4: Input="The cat sat on" → Correct next="the"
Position 5: Input="The cat sat on the" → Correct next="mat"
```

**Goal:** Maximize the probability the model assigns to the correct next token.

### Why "Next Token Prediction" Works

This simple objective is surprisingly powerful:

1. To predict "sat" after "The cat", the model must understand:
   - "cat" is a noun (subject)
   - Subjects are often followed by verbs
   - "sat" is a common verb for cats

2. To predict well, the model must learn:
   - Grammar and syntax
   - Word meanings and relationships
   - Facts about the world
   - Common patterns in language

**By learning to predict, the model learns to understand.**

---

## Part 3: The Loss Function (Measuring "Wrongness")

### What Is Loss?

Loss = a number that measures how wrong the model's prediction was.

- **Low loss** = model predicted well (correct token had high probability)
- **High loss** = model predicted poorly (correct token had low probability)

### Cross-Entropy Loss: The Standard for Classification

Cross-entropy loss is perfect for "pick one from many" tasks:

```
Model output: probability distribution over 50,000 tokens
Correct answer: one specific token (e.g., "cat")

Loss = -log(probability of correct token)
```

### Intuition: Why Negative Log?

| Probability of correct token | -log(probability) | Meaning |
|------------------------------|-------------------|---------|
| 90% (0.9) | 0.105 | Very low loss - good! |
| 50% (0.5) | 0.693 | Medium loss |
| 10% (0.1) | 2.303 | High loss - bad! |
| 1% (0.01) | 4.605 | Very high loss - very bad! |
| 0.01% (0.0001) | 9.210 | Terrible loss |

**Key properties:**
- Correct answer with high probability → low loss ✓
- Correct answer with low probability → high loss ✓
- Loss is always positive
- Loss approaches infinity as probability approaches 0

### Example Calculation

```
Model predicts next token after "The":
  "cat"    → 5%
  "dog"    → 3%
  "man"    → 2%
  "quick"  → 1%
  ... (other tokens share remaining 89%)

Correct answer: "cat"
Loss = -log(0.05) = 2.996

If model improved to give "cat" 50% probability:
Loss = -log(0.50) = 0.693  (much better!)
```

### Average Loss Over a Batch

We don't train on one token at a time. We compute average loss over a batch:

```
Batch: "The cat sat on the mat" (5 predictions)

Losses:
  "The" → "cat":  -log(0.05) = 2.996
  "The cat" → "sat": -log(0.08) = 2.526
  "The cat sat" → "on": -log(0.15) = 1.897
  "The cat sat on" → "the": -log(0.20) = 1.609
  "The cat sat on the" → "mat": -log(0.12) = 2.120

Average loss = (2.996 + 2.526 + 1.897 + 1.609 + 2.120) / 5 = 2.23
```

**Training goal: Minimize this average loss across billions of tokens.**

---

## Part 4: Updating Weights (Gradient Descent)

### The Big Picture

```
Current weights → Predictions → Loss → Gradients → Updated weights
                                         ↑
                              "Which direction reduces loss?"
```

### Gradients: The Direction to Improve

A **gradient** tells us:
- Which direction to nudge each weight
- How much that weight contributed to the error

Think of it like this:
- You're on a hill (loss landscape)
- You want to get to the bottom (minimum loss)
- The gradient points uphill
- You walk in the opposite direction (downhill)

### The Update Rule

```
new_weight = old_weight - learning_rate × gradient
```

- **Learning rate:** How big a step to take (typically 0.0001 to 0.001)
- **Gradient:** Direction and magnitude of change needed

### Why "Learning Rate" Matters

| Learning Rate | Behavior |
|---------------|----------|
| Too high (0.1) | Overshoots, loss bounces around, may never converge |
| Too low (0.0000001) | Very slow, takes forever to train |
| Just right (0.0001) | Steady improvement, finds good minimum |

Modern training uses **learning rate schedules** - start higher, decrease over time.

### Backpropagation: Computing Gradients

With 70 billion parameters, how do we compute 70 billion gradients?

**Backpropagation** uses the chain rule from calculus:

```
If loss depends on output, and output depends on weights,
then we can compute how loss depends on weights.

loss → output_layer → hidden_layer_N → ... → hidden_layer_1 → input

Work backwards, computing gradients at each step.
```

The math is complex, but frameworks (PyTorch) handle it automatically:

```python
loss.backward()  # Computes all 70 billion gradients!
optimizer.step()  # Updates all weights
```

---

## Part 5: The Training Loop in Practice

### One Training Step

```python
# 1. Get a batch of training data
batch = get_next_batch()  # e.g., 4 million tokens

# 2. Forward pass: compute predictions
predictions = model(batch.inputs)  # Run through all layers

# 3. Compute loss
loss = cross_entropy(predictions, batch.targets)

# 4. Backward pass: compute gradients
loss.backward()  # Automatic differentiation

# 5. Update weights
optimizer.step()  # Apply gradients
optimizer.zero_grad()  # Reset for next step
```

### A Full Training Run

```
Total training tokens: 15 trillion (Llama 3)
Batch size: 4 million tokens
Steps per epoch: 15T / 4M = 3.75 million steps

For each step:
  - Forward pass: ~3 seconds
  - Backward pass: ~6 seconds  
  - Total: ~10 seconds per step

Total time: 3.75M steps × 10s = ~1 year on one GPU!

With 16,000 GPUs: ~3 months
```

### What Changes During Training

| Metric | Start | Middle | End |
|--------|-------|--------|-----|
| Loss | ~11 (random) | ~3 | ~1.5 |
| Perplexity | ~60,000 | ~20 | ~4.5 |
| Next-token accuracy | ~0.002% | ~30% | ~55% |

**Perplexity** = e^loss = "how many tokens is the model confused between"
- Perplexity 4.5 means the model is "choosing between ~4-5 equally likely tokens"

---

## Part 6: The Statistics Foundation

### It's All Probability

LLMs are **probabilistic models**. Everything connects:

```
P(next_token | previous_tokens)

The model learns a conditional probability distribution.
```

### Maximum Likelihood Estimation

Training an LLM is a form of **maximum likelihood estimation**:

> "Find the parameters (weights) that make the observed training data most likely"

```
Given training text: "The cat sat"

We want weights θ that maximize:
  P("cat" | "The"; θ) × P("sat" | "The cat"; θ)

Equivalently, minimize:
  -log P("cat" | "The"; θ) - log P("sat" | "The cat"; θ)
  
This is cross-entropy loss!
```

### Why This Works: The Law of Large Numbers

With enough training data:
- The model sees every common pattern many times
- Statistics converge to true underlying distributions
- The model learns genuine language structure, not noise

```
"The cat sat on the" appears in training ~10,000 times
  → followed by "mat" ~2,000 times (20%)
  → followed by "floor" ~1,500 times (15%)
  → followed by "couch" ~800 times (8%)

The model learns these proportions!
```

### Generalization: The Key Challenge

The model sees ~15 trillion tokens, but there are infinite possible sentences.

**The model must generalize** - apply learned patterns to new, unseen text.

This works because:
1. Language has structure (grammar, semantics)
2. The model learns **patterns**, not memorization
3. Attention allows flexible composition of learned components

---

## Part 7: Engineering Scale

### The Hardware Challenge

Training a 70B parameter model requires:

| Resource | Amount |
|----------|--------|
| GPU memory | 70B params × 16 bytes = 1.1 TB (for training state) |
| GPUs | 2,000+ H100s |
| Training time | 2-3 months |
| Electricity | ~10 GWh (~$1M+ in power) |
| Total cost | $50-100+ million |

### Distributed Training

One GPU can't hold a 70B model. Solutions:

**Data Parallelism:**
- Same model on each GPU
- Different data batches
- Average gradients across GPUs

**Model Parallelism:**
- Model split across GPUs
- Each GPU holds part of the model
- Layers or attention heads distributed

**Pipeline Parallelism:**
- Different layers on different GPUs
- Data flows through the pipeline
- Like an assembly line

### From the Llama 3 Paper

```
Training infrastructure:
- 16,000 H100 GPUs
- Custom networking (RoCE)
- 405B model took ~54 days
- 15.6T training tokens

Challenges handled:
- GPU failures: ~1-2 per day across 16K GPUs
- Automatic checkpointing and restart
- Custom load balancing
```

### The Training Recipe

Modern LLM training uses many tricks:

| Technique | Purpose |
|-----------|---------|
| Mixed precision (BF16) | 2× faster, less memory |
| Gradient checkpointing | Trade compute for memory |
| Flash Attention | Faster attention, less memory |
| Learning rate warmup | Stable early training |
| Cosine schedule | Smooth learning rate decay |
| Gradient clipping | Prevent exploding gradients |
| Weight decay | Regularization, prevent overfitting |

---

## Part 8: Connecting It All

### The Full Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRE-TRAINING                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   PREPARATION (weeks)                                           │
│   ├── Crawl web data                                            │
│   ├── Clean and filter (FineWeb pipeline)                       │
│   ├── Deduplicate                                               │
│   └── Tokenize                                                  │
│                                                                 │
│   TRAINING (months)                                             │
│   For step = 1 to 3,750,000:                                    │
│   │                                                             │
│   │   1. Sample batch (4M tokens)                               │
│   │         ↓                                                   │
│   │   2. Forward pass (predict next tokens)                     │
│   │         ↓                                                   │
│   │   3. Compute cross-entropy loss                             │
│   │         ↓                                                   │
│   │   4. Backward pass (compute gradients)                      │
│   │         ↓                                                   │
│   │   5. Update weights (optimizer step)                        │
│   │         ↓                                                   │
│   │   6. Log metrics, checkpoint periodically                   │
│   │                                                             │
│   OUTPUT: Base model (next-token predictor)                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### What the Model Learned

After training, the model has:
- Encoded grammar and syntax in its weights
- Learned word meanings through context
- Stored factual knowledge
- Developed reasoning patterns
- Learned style, tone, formatting

All from the simple objective: **predict the next token**.

---

## Key Takeaways

1. **Training = making better predictions**
   - Start with random weights → end with useful weights

2. **The objective is simple**
   - Predict the next token
   - Maximize probability of correct answers

3. **Loss measures wrongness**
   - Cross-entropy: -log(probability of correct token)
   - Lower is better

4. **Gradients show how to improve**
   - Computed via backpropagation
   - Point toward lower loss

5. **Scale is massive**
   - Trillions of tokens
   - Billions of parameters
   - Thousands of GPUs
   - Months of training

6. **It's all statistics**
   - Models learn probability distributions
   - Training is maximum likelihood estimation

---

## Discussion Questions

1. Why do you think "next token prediction" leads to models that can do complex reasoning, when the training objective seems so simple?

2. If a model has low loss on training data but performs poorly on new text, what might have gone wrong?

3. Why might a model assign too much probability to common words like "the" and "a"? How might training address this?

4. What happens if we train for too long on the same data?

---

## Exercises

1. **Calculate loss:** If the model assigns 15% probability to the correct token, what's the cross-entropy loss?

2. **Estimate training time:** With 1 trillion tokens, batch size of 1M, and 5 seconds per step, how long does training take on one GPU? On 1000 GPUs?

3. **Understand perplexity:** A model has perplexity of 10. Roughly how many tokens is it "choosing between" at each prediction?
