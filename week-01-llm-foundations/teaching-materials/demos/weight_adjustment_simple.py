"""
Weight Adjustment: The Simplest Possible Example
================================================

This shows EXACTLY what "adjusting weights" means.
One weight. One input. Pure arithmetic. No magic.
"""

print("=" * 60)
print("WEIGHT ADJUSTMENT: THE SIMPLEST EXAMPLE")
print("=" * 60)

# ============================================================
# THE SETUP
# ============================================================

print("""
SETUP
-----
We have the simplest possible "neural network":
    
    Output = Weight × Input
    
That's it. ONE weight. ONE multiplication.

Goal: Find the weight that makes Output = 4 when Input = 2
      (The answer is obviously 2, but let's watch the math find it!)
""")

# Our single weight - starts at a wrong value
weight = 0.5

# Training data
input_value = 2
target = 4

learning_rate = 0.1

print(f"Starting weight: {weight}")
print(f"Input: {input_value}")
print(f"Target output: {target}")
print(f"Learning rate: {learning_rate}")

# ============================================================
# THE TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("TRAINING: WATCH THE WEIGHT CHANGE")
print("=" * 60)

print(f"\n{'Step':<6} {'Weight':<10} {'Prediction':<12} {'Loss':<12} {'Gradient':<12}")
print("-" * 60)

for step in range(10):
    # === FORWARD PASS ===
    # Use current weight to make prediction
    prediction = weight * input_value
    
    # === CALCULATE LOSS ===
    # How wrong are we? (squared error)
    loss = (prediction - target) ** 2
    
    # === CALCULATE GRADIENT ===
    # Which direction should we nudge the weight?
    # d(loss)/d(weight) = 2 * (pred - target) * input
    gradient = 2 * (prediction - target) * input_value
    
    # Print current state
    print(f"{step:<6} {weight:<10.4f} {prediction:<12.4f} {loss:<12.6f} {gradient:<12.4f}")
    
    # === UPDATE WEIGHT ===
    # Nudge the weight in the direction that reduces loss
    weight = weight - learning_rate * gradient
    
    # Stop if we're close enough
    if loss < 0.0001:
        print(f"\n✓ Converged! Final weight ≈ {weight:.4f}")
        break

print(f"""
RESULT
------
The weight automatically found the value {weight:.4f} ≈ 2.0

Check: 2.0 × 2 = 4 ✓

No human touched the weight. The math adjusted it automatically!
""")

# ============================================================
# THE KEY INSIGHT
# ============================================================

print("=" * 60)
print("THE KEY INSIGHT")
print("=" * 60)

print("""
Each step does this SIMPLE arithmetic:

    new_weight = old_weight - learning_rate × gradient

That's it!

SCALING UP TO LLMs:
- Instead of 1 weight    → 70,000,000,000 weights
- Instead of 1 input     → 4,000,000 tokens per batch  
- Instead of 10 steps    → 3,750,000 steps
- Instead of 1 multiply  → billions of matrix operations

But EVERY SINGLE WEIGHT gets updated the same way:
    
    w = w - lr × gradient

The framework computes all 70 billion gradients automatically
using the chain rule (backpropagation).

THAT'S ALL "TRAINING" IS!
""")

# ============================================================
# VISUAL: WATCHING LOSS DECREASE
# ============================================================

print("\n" + "=" * 60)
print("VISUAL: LOSS DECREASING OVER STEPS")
print("=" * 60)

# Reset and collect data for visualization
weight = 0.5
losses = []

for step in range(10):
    prediction = weight * input_value
    loss = (prediction - target) ** 2
    losses.append(loss)
    gradient = 2 * (prediction - target) * input_value
    weight = weight - learning_rate * gradient

# ASCII bar chart
print("\nLoss at each step:")
print()
max_loss = max(losses)
for i, loss in enumerate(losses):
    bar_length = int(loss / max_loss * 40)
    bar = "█" * bar_length
    print(f"Step {i}: {bar} {loss:.4f}")

print("""
The loss drops rapidly then converges to ~0.
This is exactly what happens in LLM training!
""")

# ============================================================
# INTERACTIVE EXAMPLE
# ============================================================

print("=" * 60)
print("TRY IT YOURSELF")
print("=" * 60)

print("""
Here's the complete weight update in 4 lines of Python:

```python
prediction = weight * input_value           # Forward pass
loss = (prediction - target) ** 2           # Calculate loss
gradient = 2 * (prediction - target) * input_value  # Get gradient
weight = weight - learning_rate * gradient  # Update weight
```

That's all there is to it!

For a full LLM, PyTorch does the gradient calculation automatically:

```python
prediction = model(input_batch)    # Forward pass (billions of operations)
loss = cross_entropy(prediction, target)
loss.backward()                    # Calculate ALL gradients (automatic!)
optimizer.step()                   # Update ALL weights at once
```
""")
