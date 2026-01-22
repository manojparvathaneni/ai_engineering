# Neural Networks Fundamentals - Teaching Guide

## The Big Picture

**A neural network is just a long mathematical expression with learnable parameters.**

That's it. Everything else - layers, architectures, deep learning - is just clever ways of organizing and chaining these mathematical expressions.

---

## Part 1: The Fundamental Unit

### One Neuron = One Equation

```
y = wx + b
```

Where:
- **x** = input (the data you feed in)
- **w** = weight (how much this input matters)
- **b** = bias (a baseline adjustment)
- **y** = output (the prediction)

### Concrete Example: Predicting House Price

Imagine predicting house price based on square footage:

```
price = (weight × square_feet) + bias
price = (200 × 1500) + 50000
price = $350,000
```

The **weight** (200) says "each square foot adds $200 to the price."
The **bias** (50000) says "even a 0 sq ft house has a base value of $50k" (land, location, etc.)

### Multiple Inputs: Still One Equation

What if we have square footage AND number of bedrooms?

```
price = (w1 × square_feet) + (w2 × bedrooms) + bias
price = (200 × 1500) + (10000 × 3) + 50000
price = $380,000
```

Or in vector notation:
```
y = w · x + b
```

Where **w** and **x** are now vectors (lists of numbers).

---

## Part 2: From Neurons to Layers

### A Layer = Multiple Neurons Working in Parallel

Instead of one output, what if we want multiple outputs?

```
Input: [square_feet, bedrooms, bathrooms]
       ↓
    Layer 1
       ↓
Output: [estimated_price, confidence_score, market_category]
```

Each output has its own set of weights:

```
price      = w1·x + b1
confidence = w2·x + b2  
category   = w3·x + b3
```

### Matrix Notation (Where Linear Algebra Enters)

We can express all of this compactly:

```
Y = WX + B
```

Where:
- **X** = input vector (3 values)
- **W** = weight matrix (3×3 = 9 weights)
- **B** = bias vector (3 values)
- **Y** = output vector (3 values)

**This is the "Linear Layer"** - the most fundamental building block.

---

## Part 3: Why We Need Non-Linearity

### The Problem with Pure Linear Layers

If you chain linear layers:
```
Layer 1: y1 = W1·x + B1
Layer 2: y2 = W2·y1 + B2
```

Mathematically, this collapses to:
```
y2 = W2·(W1·x + B1) + B2
y2 = (W2·W1)·x + (W2·B1 + B2)
y2 = W_combined·x + B_combined
```

**Two linear layers = one linear layer!** No matter how many you stack, you can't learn complex patterns.

### The Solution: Activation Functions

Add a non-linear function between layers:

```
Layer 1: y1 = W1·x + B1
Activation: a1 = ReLU(y1)      ← non-linear!
Layer 2: y2 = W2·a1 + B2
```

Now the math can't collapse. The network can learn complex, curved patterns.

### Common Activation Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| **ReLU** | max(0, x) | Most common, fast, works well |
| **Sigmoid** | 1/(1+e^(-x)) | Output between 0-1, good for probabilities |
| **Tanh** | (e^x - e^(-x))/(e^x + e^(-x)) | Output between -1 and 1 |
| **Softmax** | e^xi / Σe^xj | Multi-class probabilities (sums to 1) |

---

## Part 4: Common Layer Types

### 1. Linear Layer (Dense/Fully Connected)

**What it does:** Every input connects to every output
**Math:** `y = Wx + b`
**Use case:** General-purpose transformation, final prediction layers

```
Input: [x1, x2, x3]
         ↓ ↘ ↓ ↙ ↓
Output: [y1, y2]
```

### 2. Embedding Layer

**What it does:** Converts discrete IDs (like token IDs) into dense vectors
**Math:** Lookup table (matrix where row i = embedding for ID i)
**Use case:** First layer in NLP - convert tokens to vectors

```
Token ID: 42
         ↓
Embedding: [0.23, -0.45, 0.67, 0.12, ...]  (e.g., 768 dimensions)
```

### 3. Dropout Layer

**What it does:** Randomly "turns off" neurons during training
**Math:** Multiply random neurons by 0
**Use case:** Prevents overfitting (memorizing training data)

```
Training: [0.5, 0.3, 0.0, 0.8, 0.0, 0.2]  ← some zeroed out
Inference: [0.5, 0.3, 0.7, 0.8, 0.4, 0.2]  ← all active
```

### 4. Normalization Layers

**What it does:** Keeps values in a reasonable range
**Why:** Prevents numbers from exploding or vanishing as they flow through layers

**Layer Normalization** (used in transformers):
- Normalizes across features for each example
- `y = (x - mean) / std`

**Batch Normalization** (used in CNNs):
- Normalizes across examples for each feature

### 5. Convolutional Layer (CNN)

**What it does:** Slides a small "filter" across input, detecting local patterns
**Use case:** Images - finds edges, textures, shapes

```
Image patch    Filter      Output
[1 2 3]       [1 0 -1]
[4 5 6]   ×   [1 0 -1]  =  Single value
[7 8 9]       [1 0 -1]
```

### 6. Recurrent Layer (RNN/LSTM/GRU)

**What it does:** Processes sequences, maintaining "memory" of previous items
**Use case:** Sequential data (text, time series) - before transformers dominated

```
Word 1 → [hidden state] → Word 2 → [updated hidden] → Word 3 → ...
```

### 7. Attention Layer

**What it does:** Lets each position "look at" all other positions
**Use case:** Transformers! (We'll cover this in depth next)

```
"The cat sat on the mat"
     ↑___________↑
  "cat" attends to "mat" to understand the scene
```

---

## Part 5: Putting It Together - Network Architectures

### The General Pattern

```
Input
  ↓
[Layer 1] → [Activation] → [Normalization]
  ↓
[Layer 2] → [Activation] → [Normalization]
  ↓
... (repeat)
  ↓
[Output Layer]
  ↓
Prediction
```

### Example: Simple Classifier

```python
Input (784 pixels from 28×28 image)
  ↓
Linear(784 → 256) → ReLU
  ↓
Linear(256 → 128) → ReLU
  ↓
Linear(128 → 10) → Softmax
  ↓
Output (probability for each of 10 digits)
```

### Example: Transformer Block (Preview)

```python
Input tokens (embedded)
  ↓
[Self-Attention] → Add & Normalize
  ↓
[Feed-Forward (Linear → ReLU → Linear)] → Add & Normalize
  ↓
Output (same shape, but tokens now "know about" each other)
```

---

## Part 6: Training - How Parameters Learn

### The Learning Loop

1. **Forward pass:** Data flows through network, produces prediction
2. **Loss calculation:** Compare prediction to correct answer
3. **Backward pass:** Calculate how each parameter contributed to error
4. **Update:** Adjust parameters to reduce error
5. **Repeat** thousands/millions of times

### Key Concepts

**Loss Function:** Measures "how wrong" the prediction is
- MSE (Mean Squared Error) for regression
- Cross-Entropy for classification

**Gradient:** Direction to adjust each parameter
- Calculated via backpropagation (chain rule from calculus)

**Learning Rate:** How big of a step to take
- Too big: overshoots, unstable
- Too small: takes forever

**Optimizer:** Strategy for updating parameters
- SGD: simple gradient descent
- Adam: adaptive, most common today

---

## Key Takeaways for Teaching

1. **Start simple:** `y = wx + b` is the atom everything else is made of

2. **Build incrementally:** Single neuron → layer → network

3. **Non-linearity is essential:** Without it, deep networks are pointless

4. **Different layers for different jobs:** Linear for transformation, embedding for tokens, attention for relationships

5. **It's all just math:** No magic, just matrix multiplication and simple functions applied millions of times

---

## Discussion Questions

1. If neural networks are "just math," why do they seem to exhibit intelligent behavior?

2. Why might a network with 3 hidden layers learn better than one with 100 layers? (Hint: vanishing gradients)

3. How does the embedding layer relate to what we learned about tokenization?

4. Why do you think attention replaced RNNs for language tasks?

---

## Exercises

1. **Calculate by hand:** Given input [2, 3], weights [[1, 2], [3, 4]], and bias [1, 1], compute the linear layer output.

2. **Trace the shapes:** If input is 100 tokens, each embedded as 768-dimensional vectors, what's the shape after a linear layer that projects to 3072 dimensions?

3. **Design a network:** Sketch an architecture to classify movie reviews as positive/negative. What layers would you use?
