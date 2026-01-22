# Math for LLMs - No Background Required

A simple guide to the math concepts used in understanding LLMs.
No prior math background needed!

---

## 1. VECTORS = Lists of Numbers

```
A vector is just a list of numbers in a row:

"cat" → [0.23, -0.45, 0.67, 0.12, 0.89]
         ↑     ↑      ↑     ↑     ↑
         Just 5 numbers. That's a vector.
```

**In LLMs:** Each word becomes a vector (list of 768-4096 numbers).
Why? One number can't capture meaning. Many numbers together can.

---

## 2. MATRICES = Grids of Numbers

```
A matrix is numbers in rows and columns:

     [0.2,  0.5,  0.1]    ← row 0
     [0.9,  0.1,  0.4]    ← row 1
     [0.3,  0.7,  0.2]    ← row 2
       ↑     ↑     ↑
      col   col   col
       0     1     2

This is a 3×3 matrix (3 rows, 3 columns).
```

**In LLMs:** Weight matrices store all the learned numbers.
A 70B parameter model = matrices containing 70 billion numbers total.

---

## 3. MATRIX MULTIPLICATION = Weighted Combinations

The core operation of neural networks. Each output is a weighted sum of inputs.

```
Input: [2, 3]  (a vector with 2 numbers)

Weights:       (a 3×2 matrix)
  [1, 4]
  [2, 5]
  [3, 6]

Output:
  out[0] = 2×1 + 3×4 = 2 + 12 = 14
  out[1] = 2×2 + 3×5 = 4 + 15 = 19
  out[2] = 2×3 + 3×6 = 6 + 18 = 24

Result: [14, 19, 24]  (a vector with 3 numbers)
```

**Pattern:** Multiply matching positions, add them up.

**In LLMs:** Every layer does this: `output = input × weights + bias`

---

## 4. DOT PRODUCT = Similarity Score

Take two vectors, multiply matching positions, sum → one number.

```
A = [1, 2, 3]
B = [4, 5, 6]

A · B = 1×4 + 2×5 + 3×6
      = 4 + 10 + 18
      = 32
```

**Key insight:** High dot product = similar vectors!

```
Same direction:  [1,0] · [1,0] = 1    (maximum similarity)
Opposite:        [1,0] · [-1,0] = -1  (opposite)
Perpendicular:   [1,0] · [0,1] = 0    (unrelated)
```

**In LLMs:** Attention uses Q·K to measure "how relevant is this word to that word?"

---

## 5. GRADIENT = Direction to Improve

No calculus needed! Just think of hills.

```
You're on a hill. You want to find the lowest point.
Which way do you walk? DOWNHILL.

The gradient tells you: "The hill slopes THIS direction"
You walk: the OPPOSITE direction (down, not up)
```

**For a weight:**
```
Gradient = +0.5 → "Increasing this weight makes things WORSE"
                → So DECREASE it

Gradient = -0.3 → "Increasing this weight makes things BETTER"  
                → So INCREASE it
```

**The update rule:**
```
new_weight = old_weight - learning_rate × gradient
                         ↑
                The minus sign means "go opposite direction"
```

**In LLMs:** Backpropagation computes gradients for ALL 70 billion weights automatically. We just call `loss.backward()`.

---

## 6. PROBABILITY = Percentages (Must Sum to 100%)

```
Model predicts next word:

  "the"   → 25%
  "mat"   → 20%
  "floor" → 15%
  "cat"   → 10%
  ...     → 30%
  ─────────────
  Total:    100%  ← MUST sum to 1 (or 100%)
```

**Key properties:**
- Each probability is between 0% and 100%
- All probabilities sum to exactly 100%
- Higher probability = model thinks this is more likely

**In LLMs:** The output layer produces a probability for every word in vocabulary (50,000+ words, each with a probability, summing to 100%).

---

## 7. SOFTMAX = Convert Anything to Probabilities

Raw model outputs (called "logits") can be any numbers: negative, huge, whatever.
Softmax converts them to valid probabilities.

```
Raw scores (logits): [2.0, 1.0, 0.5]

Step 1: Apply e^x to each
  e^2.0 = 7.39
  e^1.0 = 2.72
  e^0.5 = 1.65

Step 2: Divide each by sum
  sum = 7.39 + 2.72 + 1.65 = 11.76
  
  7.39/11.76 = 0.63 (63%)
  2.72/11.76 = 0.23 (23%)
  1.65/11.76 = 0.14 (14%)
                     ────
                     100% ✓
```

**Properties:**
- Biggest input → biggest probability (preserves ranking)
- All outputs are positive (0% to 100%)
- Always sums to 100%

**In LLMs:** 
- Attention weights: softmax(QK^T/√d)
- Output probabilities: softmax(logits)

---

## 8. LOGARITHM = Opposite of Exponents

```
If:   2³ = 8
Then: log₂(8) = 3

"What power do I raise 2 to, to get 8? Answer: 3"
```

**The one key thing to know for LLMs:**

```
log of a number between 0 and 1 is NEGATIVE

log(0.9)  = -0.1   (high prob → small negative)
log(0.5)  = -0.7   (medium prob → medium negative)
log(0.1)  = -2.3   (low prob → big negative)
log(0.01) = -4.6   (tiny prob → huge negative)
```

**In LLMs:** Cross-entropy loss = -log(probability of correct answer)

```
If model gave correct answer 90% probability:
  Loss = -log(0.9) = 0.1  (low loss = good!)
  
If model gave correct answer 1% probability:
  Loss = -log(0.01) = 4.6  (high loss = bad!)
```

The negative sign flips it so higher probability = lower loss.

---

## PUTTING IT ALL TOGETHER

Here's how these concepts flow in one forward pass:

```
"The cat sat" (text)
      ↓
  Tokenize (no math, just lookup)
      ↓
[1, 42, 89] (token IDs)
      ↓
  Embedding lookup → VECTORS
      ↓
[[0.2, 0.5, ...], [0.9, 0.1, ...], [0.3, 0.7, ...]]
      ↓
  Attention: Q·K (DOT PRODUCT for similarity)
      ↓
  Softmax → attention PROBABILITIES
      ↓
  MATRIX MULTIPLY with values
      ↓
  More MATRIX MULTIPLIES through layers
      ↓
  Final SOFTMAX → word PROBABILITIES
      ↓
[0.02, 0.15, 0.05, ...] (one prob per word in vocab)
      ↓
  Pick next word: "on" (highest probability)
```

**For training, add:**
```
  Compare to correct answer
      ↓
  Loss = -LOG(probability of correct)
      ↓
  Compute GRADIENTS (automatic!)
      ↓
  Update weights: w = w - lr × gradient
```

---

## Quick Reference

| Concept | One-Liner |
|---------|-----------|
| Vector | List of numbers |
| Matrix | Grid of numbers |
| Matrix multiply | Weighted combination |
| Dot product | Similarity score (multiply & sum) |
| Gradient | "Which direction to nudge" |
| Probability | Percentage (sums to 100%) |
| Softmax | Converts scores to probabilities |
| Logarithm | log(small) = big negative |

---

## The Good News

**You don't need to compute any of this by hand!**

- PyTorch/TensorFlow handle all the math
- `loss.backward()` computes ALL gradients automatically
- You just need to understand WHAT these operations do, not HOW to calculate them

Understanding the intuition (this guide) is enough to understand LLMs!
