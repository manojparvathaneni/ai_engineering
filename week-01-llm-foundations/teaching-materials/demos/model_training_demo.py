"""
Model Training Deep Dive - Teaching Demo
========================================

This demo walks through training concepts with actual numbers:
1. Why training matters (random vs trained)
2. Cross-entropy loss calculation
3. The training loop step-by-step
4. Gradient descent intuition
5. Training at scale

All with pure Python/numpy - see exactly what happens!
"""

import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("=" * 70)
print("MODEL TRAINING DEEP DIVE")
print("=" * 70)


# =============================================================================
# PART 1: BEFORE AND AFTER TRAINING
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: WHY TRAINING MATTERS")
print("=" * 70)

print("""
Before training: Model weights are RANDOM
After training:  Model weights encode LANGUAGE PATTERNS

Let's see the difference in predictions...
""")

# Simulate a tiny vocabulary
vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "under", "table", "floor"]
vocab_size = len(vocab)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# UNTRAINED: Random logits → roughly uniform probabilities
np.random.seed(42)
untrained_logits = np.random.randn(vocab_size) * 0.1  # Small random values

print("UNTRAINED MODEL (random weights)")
print("-" * 50)
print("Input: 'The cat sat on the' → Predict next word")
print()
untrained_probs = softmax(untrained_logits)
print("Predictions (roughly uniform - no learning yet):")
for word, prob in sorted(zip(vocab, untrained_probs), key=lambda x: -x[1])[:5]:
    print(f"  '{word}': {prob*100:.2f}%")
print(f"  ... (all ~{100/vocab_size:.1f}% each)")

# TRAINED: Logits reflect learned patterns
# "mat" should be most likely after "The cat sat on the"
trained_logits = np.array([
    -1.0,   # "the" - unlikely (already said "the")
    -2.0,   # "cat" - unlikely
    -2.5,   # "sat" - unlikely
    -2.0,   # "on" - unlikely
    2.5,    # "mat" - VERY likely!
    -1.5,   # "dog" - unlikely
    -2.0,   # "ran" - unlikely
    -1.0,   # "under" - possible but unlikely
    -0.5,   # "table" - somewhat possible
    1.5,    # "floor" - also likely
])

print("\n\nTRAINED MODEL (learned weights)")
print("-" * 50)
print("Input: 'The cat sat on the' → Predict next word")
print()
trained_probs = softmax(trained_logits)
print("Predictions (learned from billions of sentences):")
for word, prob in sorted(zip(vocab, trained_probs), key=lambda x: -x[1])[:5]:
    print(f"  '{word}': {prob*100:.2f}%")

print("\n✓ Training transformed random guesses into meaningful predictions!")


# =============================================================================
# PART 2: CROSS-ENTROPY LOSS
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: CROSS-ENTROPY LOSS")
print("=" * 70)

print("""
Loss measures "how wrong" the prediction was.
Cross-entropy loss = -log(probability of correct answer)

Properties:
  - Correct answer with HIGH probability → LOW loss (good!)
  - Correct answer with LOW probability → HIGH loss (bad!)
""")

def cross_entropy_loss(probs, correct_idx):
    """Compute cross-entropy loss for a single prediction."""
    correct_prob = probs[correct_idx]
    loss = -np.log(correct_prob)
    return loss, correct_prob

# The correct answer is "mat" (index 4)
correct_idx = vocab.index("mat")

print("\nExample: Correct next word is 'mat'")
print("-" * 50)

# Untrained model
loss_untrained, prob_untrained = cross_entropy_loss(untrained_probs, correct_idx)
print(f"\nUNTRAINED model:")
print(f"  P('mat') = {prob_untrained:.4f} ({prob_untrained*100:.2f}%)")
print(f"  Loss = -log({prob_untrained:.4f}) = {loss_untrained:.4f}")

# Trained model
loss_trained, prob_trained = cross_entropy_loss(trained_probs, correct_idx)
print(f"\nTRAINED model:")
print(f"  P('mat') = {prob_trained:.4f} ({prob_trained*100:.2f}%)")
print(f"  Loss = -log({prob_trained:.4f}) = {loss_trained:.4f}")

print(f"\nImprovement: {loss_untrained:.2f} → {loss_trained:.2f} ({(1-loss_trained/loss_untrained)*100:.0f}% reduction)")


# Show the loss curve
print("\n\nLOSS vs PROBABILITY (the relationship)")
print("-" * 50)
print(f"{'Probability':<15} {'Loss':<15} {'Interpretation'}")
print("-" * 50)

test_probs = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
for p in test_probs:
    loss = -np.log(p)
    if p < 0.05:
        interp = "Terrible prediction"
    elif p < 0.20:
        interp = "Poor prediction"
    elif p < 0.50:
        interp = "Okay prediction"
    elif p < 0.80:
        interp = "Good prediction"
    else:
        interp = "Excellent prediction"
    print(f"{p:<15.3f} {loss:<15.4f} {interp}")


# =============================================================================
# PART 3: BATCH LOSS CALCULATION
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: BATCH LOSS CALCULATION")
print("=" * 70)

print("""
In practice, we compute loss over a BATCH of predictions.
Each position in the text gives us one prediction to evaluate.
""")

# Training example: "The cat sat on the mat"
# At each position, predict the next word
training_sequence = [
    ("The", "cat"),
    ("The cat", "sat"),
    ("The cat sat", "on"),
    ("The cat sat on", "the"),
    ("The cat sat on the", "mat"),
]

print("Training sequence: 'The cat sat on the mat'")
print("-" * 60)
print()

# Simulate model predictions at each position
# (In reality, these come from the neural network)
np.random.seed(123)
position_losses = []

print(f"{'Context':<25} {'Target':<8} {'P(target)':<12} {'Loss'}")
print("-" * 60)

for context, target in training_sequence:
    # Simulate model giving some probability to target
    # (trained model does better on common patterns)
    if target in ["the", "on"]:
        target_prob = np.random.uniform(0.15, 0.30)  # Common words
    elif target in ["cat", "mat"]:
        target_prob = np.random.uniform(0.05, 0.15)  # Context-dependent
    else:
        target_prob = np.random.uniform(0.03, 0.10)  # Less predictable
    
    loss = -np.log(target_prob)
    position_losses.append(loss)
    
    print(f"'{context}'  →  '{target}'      {target_prob:.3f}        {loss:.4f}")

avg_loss = np.mean(position_losses)
print("-" * 60)
print(f"{'AVERAGE LOSS:':<47} {avg_loss:.4f}")

# Convert to perplexity
perplexity = np.exp(avg_loss)
print(f"{'PERPLEXITY:':<47} {perplexity:.2f}")
print()
print(f"Perplexity {perplexity:.1f} means the model is 'choosing between'")
print(f"~{perplexity:.0f} equally likely words at each position.")


# =============================================================================
# PART 4: GRADIENT DESCENT INTUITION
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: GRADIENT DESCENT INTUITION")
print("=" * 70)

print("""
Gradient descent finds weights that minimize loss.

Think of it like finding the lowest point in a valley:
  1. You're standing somewhere (current weights)
  2. Check which way is downhill (compute gradient)
  3. Take a step downhill (update weights)
  4. Repeat until you reach the bottom (minimum loss)
""")

# Simple 1D example: finding the minimum of a parabola
print("\nSimple example: Find x that minimizes f(x) = (x - 3)²")
print("-" * 50)
print("The answer is x = 3 (minimum = 0)")
print()

def f(x):
    return (x - 3) ** 2

def gradient_f(x):
    return 2 * (x - 3)

# Start at a random point
x = 10.0
learning_rate = 0.1

print(f"Starting at x = {x}")
print(f"Learning rate = {learning_rate}")
print()
print(f"{'Step':<6} {'x':<12} {'f(x)':<12} {'gradient':<12} {'step'}")
print("-" * 55)

for step in range(10):
    current_loss = f(x)
    grad = gradient_f(x)
    step_size = learning_rate * grad
    
    print(f"{step:<6} {x:<12.4f} {current_loss:<12.4f} {grad:<12.4f} {-step_size:+.4f}")
    
    # Update x
    x = x - learning_rate * grad

print()
print(f"After 10 steps: x = {x:.4f}, f(x) = {f(x):.6f}")
print("We found the minimum!")


# =============================================================================
# PART 5: NEURAL NETWORK TRAINING SIMULATION
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: MINI NEURAL NETWORK TRAINING")
print("=" * 70)

print("""
Let's train a TINY neural network to predict the next token.
This shows the complete training loop with real numbers.

Network: Input (3) → Hidden (4) → Output (3)
Task: Learn to output [0, 1, 0] when input is [1, 0, 0]
      (Like learning "The" → "cat")
""")

class TinyNetwork:
    def __init__(self):
        # Random initialization
        np.random.seed(42)
        self.W1 = np.random.randn(3, 4) * 0.5  # Input → Hidden
        self.b1 = np.zeros(4)
        self.W2 = np.random.randn(4, 3) * 0.5  # Hidden → Output
        self.b2 = np.zeros(3)
        
    def forward(self, x):
        # Hidden layer with ReLU
        self.z1 = x @ self.W1 + self.b1
        self.h = np.maximum(0, self.z1)  # ReLU
        
        # Output layer with softmax
        self.z2 = self.h @ self.W2 + self.b2
        self.probs = softmax(self.z2)
        
        return self.probs
    
    def backward(self, x, target_idx):
        # Gradient of cross-entropy + softmax
        dz2 = self.probs.copy()
        dz2[target_idx] -= 1  # Derivative of CE loss w.r.t. logits
        
        # Gradients for W2, b2
        dW2 = np.outer(self.h, dz2)
        db2 = dz2
        
        # Backprop through hidden layer
        dh = dz2 @ self.W2.T
        dz1 = dh * (self.z1 > 0)  # ReLU derivative
        
        # Gradients for W1, b1
        dW1 = np.outer(x, dz1)
        db1 = dz1
        
        return dW1, db1, dW2, db2
    
    def update(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
    
    def count_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

# Training data
x = np.array([1.0, 0.0, 0.0])  # Input: "The" (one-hot)
target_idx = 1                   # Target: "cat" (index)

print(f"Input (one-hot for 'The'): {x}")
print(f"Target: index {target_idx} ('cat')")
print()

# Create network
net = TinyNetwork()
print(f"Network parameters: {net.count_params()}")
print()

# Training loop
print("TRAINING LOOP")
print("-" * 60)
print(f"{'Step':<6} {'Loss':<12} {'P(cat)':<12} {'Status'}")
print("-" * 60)

learning_rate = 0.5
losses = []

for step in range(20):
    # Forward pass
    probs = net.forward(x)
    
    # Compute loss
    loss = -np.log(probs[target_idx])
    losses.append(loss)
    
    # Determine status
    if probs[target_idx] > 0.9:
        status = "✓ Excellent!"
    elif probs[target_idx] > 0.5:
        status = "Good"
    elif probs[target_idx] > 0.3:
        status = "Learning..."
    else:
        status = "Still learning"
    
    if step % 2 == 0 or step == 19:
        print(f"{step:<6} {loss:<12.4f} {probs[target_idx]:<12.4f} {status}")
    
    # Backward pass
    grads = net.backward(x, target_idx)
    
    # Update weights
    net.update(grads, learning_rate)

print("-" * 60)
print()
print(f"Final predictions: {net.forward(x)}")
print(f"Probability of 'cat': {net.forward(x)[target_idx]:.4f}")
print()
print("✓ The network learned to predict 'cat' after 'The'!")


# =============================================================================
# PART 6: TRAINING AT SCALE
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: TRAINING AT SCALE")
print("=" * 70)

print("""
Real LLM training is this same loop, but MASSIVE in scale.
Let's calculate what it takes to train Llama 3 70B...
""")

# Llama 3 70B training stats
params = 70e9  # 70 billion parameters
training_tokens = 15e12  # 15 trillion tokens
batch_size = 4e6  # 4 million tokens per batch

# Calculate training steps
total_steps = training_tokens / batch_size

# Time estimates (approximate)
forward_time = 3  # seconds
backward_time = 6  # seconds (usually ~2x forward)
step_time = forward_time + backward_time + 1  # +1 for overhead

total_time_single_gpu = total_steps * step_time  # seconds
total_time_hours = total_time_single_gpu / 3600
total_time_days = total_time_hours / 24
total_time_years = total_time_days / 365

print("LLAMA 3 70B TRAINING")
print("-" * 60)
print(f"Parameters:        {params/1e9:.0f} billion")
print(f"Training tokens:   {training_tokens/1e12:.0f} trillion")
print(f"Batch size:        {batch_size/1e6:.0f} million tokens")
print(f"Training steps:    {total_steps/1e6:.2f} million")
print()
print(f"Time per step:     ~{step_time} seconds")
print(f"Total (1 GPU):     {total_time_years:.1f} years (!)")
print()

# With distributed training
n_gpus = 16000
distributed_time_days = total_time_days / n_gpus
# Add overhead for communication (~20%)
distributed_time_days *= 1.2

print(f"With {n_gpus:,} GPUs:")
print(f"  Training time:   ~{distributed_time_days:.0f} days")
print()

# Cost estimates
gpu_hour_cost = 3  # $ per H100 hour (rough estimate)
total_gpu_hours = n_gpus * distributed_time_days * 24
total_cost = total_gpu_hours * gpu_hour_cost

print(f"COST ESTIMATE")
print("-" * 60)
print(f"GPU hours:         {total_gpu_hours/1e6:.1f} million")
print(f"At ${gpu_hour_cost}/GPU-hour: ${total_cost/1e6:.0f} million")
print()

# Memory requirements
bytes_per_param_training = 16  # 2 (weights) + 2 (gradients) + 8 (optimizer) + 4 (activations)
memory_gb = params * bytes_per_param_training / 1e9

print(f"MEMORY REQUIREMENTS")
print("-" * 60)
print(f"Per parameter:     ~{bytes_per_param_training} bytes (training)")
print(f"Total memory:      ~{memory_gb:.0f} GB")
print(f"H100 memory:       80 GB")
print(f"GPUs needed:       {int(np.ceil(memory_gb / 80))}+ (with model parallelism)")


# =============================================================================
# PART 7: THE LEARNING CURVE
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: THE LEARNING CURVE")
print("=" * 70)

print("""
Loss decreases over training, following a characteristic curve.
Let's visualize what training progress looks like...
""")

# Simulate a training loss curve
np.random.seed(42)

# Typical loss curve: starts high, drops quickly, then slowly improves
steps = np.arange(0, 3750000, 10000)  # 3.75M steps
base_loss = 11 * np.exp(-steps / 500000) + 1.5  # Exponential decay to ~1.5
noise = np.random.randn(len(steps)) * 0.1  # Add some noise
training_loss = base_loss + noise

print("Training Loss Over Time (ASCII plot)")
print("-" * 60)

# ASCII plot
def ascii_plot(values, width=50, height=15):
    """Create a simple ASCII plot."""
    min_val = min(values)
    max_val = max(values)
    
    # Sample values to fit width
    indices = np.linspace(0, len(values)-1, width).astype(int)
    sampled = [values[i] for i in indices]
    
    # Create plot
    for row in range(height):
        threshold = max_val - (row / (height - 1)) * (max_val - min_val)
        line = ""
        for val in sampled:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        # Add y-axis label
        if row == 0:
            label = f"{max_val:.1f}"
        elif row == height - 1:
            label = f"{min_val:.1f}"
        elif row == height // 2:
            label = f"{(max_val + min_val)/2:.1f}"
        else:
            label = "    "
        print(f"{label:>5} │{line}│")
    
    print(f"      └{'─' * width}┘")
    print(f"       Step 0{' ' * (width - 8)}Step 3.75M")

ascii_plot(training_loss.tolist())

print()
print("Key observations:")
print("  - Loss starts high (~11) with random weights")
print("  - Drops quickly in early training (easy patterns)")
print("  - Improvement slows down (diminishing returns)")
print("  - Converges to ~1.5 (fundamental uncertainty in language)")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
MODEL TRAINING - KEY CONCEPTS

1. PURPOSE: Turn random weights into useful weights
   - Untrained: random predictions
   - Trained: meaningful predictions

2. OBJECTIVE: Predict the next token
   - Simple but powerful
   - Forces the model to learn language

3. LOSS FUNCTION: Cross-entropy
   - Loss = -log(probability of correct token)
   - High probability → low loss (good)
   - Low probability → high loss (bad)

4. OPTIMIZATION: Gradient descent
   - Compute gradient: "which way reduces loss?"
   - Update weights: step in that direction
   - Repeat millions of times

5. THE TRAINING LOOP:
   for step in range(millions):
       predictions = model(batch)      # Forward pass
       loss = cross_entropy(predictions, targets)
       gradients = loss.backward()     # Backward pass
       optimizer.step(gradients)       # Update weights

6. SCALE (for 70B model):
   - 15 trillion training tokens
   - 16,000 GPUs
   - ~2 months training
   - ~$50-100 million cost

7. STATISTICS FOUNDATION:
   - LLMs learn probability distributions
   - Training = maximum likelihood estimation
   - "Find weights that make training data most likely"
""")


# Quick reference formulas
print("\n" + "-" * 70)
print("QUICK REFERENCE FORMULAS")
print("-" * 70)
print("""
Cross-entropy loss (single prediction):
    L = -log(P_correct)

Cross-entropy loss (batch):
    L = -(1/N) × Σ log(P_correct_i)

Perplexity:
    PPL = e^L
    (Lower is better; ~1 would be perfect prediction)

Gradient descent update:
    θ_new = θ_old - learning_rate × gradient

Training steps:
    steps = training_tokens / batch_size

Training time (distributed):
    time ≈ (steps × time_per_step) / n_gpus
""")
