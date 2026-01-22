"""
Mini Pre-Training Simulation
============================

This demo shows EXACTLY what happens during pre-training:
1. Weights start as random numbers
2. We make predictions using these numbers
3. We measure how wrong we are (loss)
4. We calculate which direction to nudge each number (gradient)
5. We update each number slightly
6. Repeat until predictions are good!

This is a COMPLETE working language model, just tiny:
- Vocabulary: 10 words
- Model: ~200 parameters (vs 70 billion in Llama)
- Training data: 5 sentences (vs 15 trillion tokens)

But the PROCESS is identical to real pre-training!
"""

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=100)
np.random.seed(42)

print("=" * 70)
print("MINI PRE-TRAINING SIMULATION")
print("=" * 70)


# =============================================================================
# PART 1: SETUP - OUR TINY LANGUAGE
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: OUR TINY LANGUAGE")
print("=" * 70)

# Our vocabulary - just 10 words
VOCAB = ["<PAD>", "the", "cat", "dog", "sat", "ran", "on", "mat", "floor", "fast"]
VOCAB_SIZE = len(VOCAB)
word_to_id = {w: i for i, w in enumerate(VOCAB)}
id_to_word = {i: w for i, w in enumerate(VOCAB)}

print(f"\nVocabulary ({VOCAB_SIZE} words):")
for i, word in enumerate(VOCAB):
    print(f"  {i}: '{word}'")

# Our training data - 5 simple sentences
TRAINING_DATA = [
    "the cat sat on the mat",
    "the dog ran on the floor", 
    "the cat ran fast",
    "the dog sat on the mat",
    "the cat sat on the floor",
]

print(f"\nTraining data ({len(TRAINING_DATA)} sentences):")
for sent in TRAINING_DATA:
    print(f"  '{sent}'")


# =============================================================================
# PART 2: THE MODEL - JUST NUMBERS IN MATRICES
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: THE MODEL (JUST NUMBERS!)")
print("=" * 70)

print("""
Our model has these weight matrices:
  1. Embedding: converts word ID → vector (10 × 8 = 80 numbers)
  2. Hidden layer: transforms vectors (8 × 16 = 128 numbers)  
  3. Output layer: predicts next word (16 × 10 = 160 numbers)

Total: ~368 parameters (just 368 numbers we need to tune!)
""")

class TinyLM:
    """A tiny language model - shows exactly what's inside."""
    
    def __init__(self, vocab_size=10, embed_dim=8, hidden_dim=16):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights RANDOMLY - this is the starting point
        # Xavier/He initialization scaled down for stability
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.3
        self.W_hidden = np.random.randn(embed_dim, hidden_dim) * 0.3
        self.b_hidden = np.zeros(hidden_dim)
        self.W_output = np.random.randn(hidden_dim, vocab_size) * 0.3
        self.b_output = np.zeros(vocab_size)
        
        # Count parameters
        self.n_params = (vocab_size * embed_dim +      # embedding
                        embed_dim * hidden_dim + hidden_dim +  # hidden layer
                        hidden_dim * vocab_size + vocab_size)  # output layer
    
    def show_weights(self, name=""):
        """Display the actual numbers in our weight matrices."""
        print(f"\n{'='*60}")
        print(f"WEIGHT VALUES {name}")
        print(f"{'='*60}")
        
        print(f"\nEmbedding matrix ({self.vocab_size} × {self.embed_dim} = {self.vocab_size * self.embed_dim} numbers):")
        print("Each row is one word's vector representation")
        for i in range(min(4, self.vocab_size)):
            print(f"  '{VOCAB[i]:>5}' (id={i}): {self.embedding[i]}")
        print("  ...")
        
        print(f"\nHidden layer weights ({self.embed_dim} × {self.hidden_dim} = {self.embed_dim * self.hidden_dim} numbers):")
        print(f"  First row: {self.W_hidden[0][:8]}...")
        print(f"  (+ {self.hidden_dim} bias terms)")
        
        print(f"\nOutput layer weights ({self.hidden_dim} × {self.vocab_size} = {self.hidden_dim * self.vocab_size} numbers):")
        print(f"  First row: {self.W_output[0]}")
        print(f"  (+ {self.vocab_size} bias terms)")
        
        print(f"\nTotal parameters: {self.n_params}")
    
    def forward(self, token_ids):
        """
        Forward pass - use the numbers to make a prediction.
        
        Input: list of token IDs [1, 2, 4] for "the cat sat"
        Output: probability distribution over next word
        """
        # Store for backward pass
        self.input_ids = token_ids
        
        # Step 1: Look up embeddings (just indexing into a matrix)
        self.embeddings = self.embedding[token_ids]  # (seq_len, embed_dim)
        
        # Step 2: Average the embeddings (simple way to combine)
        self.avg_embedding = np.mean(self.embeddings, axis=0)  # (embed_dim,)
        
        # Step 3: Hidden layer: multiply by weights, add bias, apply ReLU
        self.hidden_input = self.avg_embedding @ self.W_hidden + self.b_hidden
        self.hidden = np.maximum(0, self.hidden_input)  # ReLU activation
        
        # Step 4: Output layer: multiply by weights, add bias
        self.logits = self.hidden @ self.W_output + self.b_output
        
        # Step 5: Softmax to get probabilities
        exp_logits = np.exp(self.logits - np.max(self.logits))
        self.probs = exp_logits / exp_logits.sum()
        
        return self.probs
    
    def backward(self, target_id, learning_rate=0.1):
        """
        Backward pass - calculate gradients and update weights.
        
        This shows EXACTLY how each number gets adjusted!
        """
        # Gradient of loss w.r.t. logits (cross-entropy + softmax)
        d_logits = self.probs.copy()
        d_logits[target_id] -= 1  # This is the key derivative!
        
        # Gradient for output layer
        d_W_output = np.outer(self.hidden, d_logits)
        d_b_output = d_logits
        
        # Gradient through hidden layer
        d_hidden = d_logits @ self.W_output.T
        d_hidden_input = d_hidden * (self.hidden_input > 0)  # ReLU gradient
        
        # Gradient for hidden layer
        d_W_hidden = np.outer(self.avg_embedding, d_hidden_input)
        d_b_hidden = d_hidden_input
        
        # Gradient for embeddings
        d_avg_embedding = d_hidden_input @ self.W_hidden.T
        
        # Gradient for each input embedding
        d_embeddings = d_avg_embedding / len(self.input_ids)
        
        # UPDATE THE WEIGHTS (this is the key part!)
        # new_value = old_value - learning_rate * gradient
        self.W_output -= learning_rate * d_W_output
        self.b_output -= learning_rate * d_b_output
        self.W_hidden -= learning_rate * d_W_hidden
        self.b_hidden -= learning_rate * d_b_hidden
        
        # Update embeddings for input tokens
        for token_id in self.input_ids:
            self.embedding[token_id] -= learning_rate * d_embeddings
        
        return d_W_output, d_W_hidden  # Return for visualization


# Create our tiny model
model = TinyLM()
model.show_weights("(BEFORE TRAINING - RANDOM)")


# =============================================================================
# PART 3: ONE TRAINING STEP IN DETAIL
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: ONE TRAINING STEP (DETAILED)")
print("=" * 70)

print("""
Let's trace through ONE training step in complete detail.
We'll see exactly which numbers change and why.
""")

# Training example: "the cat sat" → should predict "on"
input_text = "the cat sat"
target_text = "on"

input_ids = [word_to_id[w] for w in input_text.split()]
target_id = word_to_id[target_text]

print(f"Training example:")
print(f"  Input: '{input_text}' → token IDs: {input_ids}")
print(f"  Target: '{target_text}' → token ID: {target_id}")

# Store old weights to show the change
old_W_output_sample = model.W_output[0, :5].copy()
old_embedding_sample = model.embedding[1, :4].copy()  # "the"

print(f"\n--- FORWARD PASS ---")
print(f"Using current weights to make a prediction...")

probs = model.forward(input_ids)

print(f"\nPredictions (probability of each word being next):")
sorted_probs = sorted(enumerate(probs), key=lambda x: -x[1])
for idx, prob in sorted_probs[:5]:
    marker = " ← TARGET" if idx == target_id else ""
    print(f"  '{id_to_word[idx]}': {prob*100:.2f}%{marker}")
print(f"  ...")

# Calculate loss
loss = -np.log(probs[target_id])
print(f"\nLoss = -log(P('{target_text}')) = -log({probs[target_id]:.4f}) = {loss:.4f}")
print(f"(Lower is better. Random would be ~{-np.log(1/VOCAB_SIZE):.2f})")

print(f"\n--- BACKWARD PASS ---")
print(f"Calculating gradients (which direction to nudge each weight)...")

d_W_output, d_W_hidden = model.backward(target_id, learning_rate=0.5)

print(f"\nGradient for output layer (first few values):")
print(f"  {d_W_output[0, :5]}")
print(f"  (Negative = increase weight, Positive = decrease weight)")

print(f"\n--- WEIGHT UPDATE ---")
print(f"Applying: new_weight = old_weight - learning_rate × gradient")

new_W_output_sample = model.W_output[0, :5]
new_embedding_sample = model.embedding[1, :4]

print(f"\nOutput layer weights (first 5 of first row):")
print(f"  Before: {old_W_output_sample}")
print(f"  After:  {new_W_output_sample}")
print(f"  Change: {new_W_output_sample - old_W_output_sample}")

print(f"\nEmbedding for 'the' (first 4 values):")
print(f"  Before: {old_embedding_sample}")
print(f"  After:  {new_embedding_sample}")
print(f"  Change: {new_embedding_sample - old_embedding_sample}")

print("\n✓ Each number was nudged slightly to make 'on' more likely next time!")


# =============================================================================
# PART 4: WATCH THE NUMBERS CHANGE OVER TRAINING
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: WATCHING WEIGHTS CHANGE OVER TRAINING")
print("=" * 70)

print("""
Now let's train for multiple steps and watch:
1. The numbers (weights) changing
2. The loss decreasing
3. The predictions improving
""")

# Reset model
model = TinyLM()

# Prepare all training examples
def prepare_training_data(sentences):
    """Convert sentences to (input_ids, target_id) pairs."""
    examples = []
    for sent in sentences:
        words = sent.split()
        for i in range(1, len(words)):
            input_ids = [word_to_id[w] for w in words[:i]]
            target_id = word_to_id[words[i]]
            examples.append((input_ids, target_id, ' '.join(words[:i]), words[i]))
    return examples

training_examples = prepare_training_data(TRAINING_DATA)
print(f"\nTraining examples ({len(training_examples)} total):")
for inp_ids, tgt_id, inp_text, tgt_text in training_examples[:5]:
    print(f"  '{inp_text}' → '{tgt_text}'")
print(f"  ... and {len(training_examples) - 5} more")

# Track specific weights to show how they change
weight_history = []
loss_history = []

# Track the embedding for "cat"
cat_id = word_to_id["cat"]

print(f"\n" + "-" * 60)
print("TRAINING PROGRESS")
print("-" * 60)
print(f"{'Epoch':<8} {'Avg Loss':<12} {'cat embed[0]':<15} {'W_out[0,0]':<15}")
print("-" * 60)

# Training loop
n_epochs = 50
learning_rate = 0.3

for epoch in range(n_epochs):
    epoch_losses = []
    
    # Shuffle training examples
    np.random.shuffle(training_examples)
    
    for input_ids, target_id, _, _ in training_examples:
        # Forward pass
        probs = model.forward(input_ids)
        loss = -np.log(probs[target_id] + 1e-10)
        epoch_losses.append(loss)
        
        # Backward pass (updates weights)
        model.backward(target_id, learning_rate)
    
    avg_loss = np.mean(epoch_losses)
    loss_history.append(avg_loss)
    weight_history.append({
        'cat_embed_0': model.embedding[cat_id, 0],
        'W_out_0_0': model.W_output[0, 0]
    })
    
    if epoch % 5 == 0 or epoch == n_epochs - 1:
        print(f"{epoch:<8} {avg_loss:<12.4f} {model.embedding[cat_id, 0]:<15.4f} {model.W_output[0, 0]:<15.4f}")

# Learning rate decay for final fine-tuning
learning_rate = 0.1
for epoch in range(20):
    for input_ids, target_id, _, _ in training_examples:
        probs = model.forward(input_ids)
        loss = -np.log(probs[target_id] + 1e-10)
        model.backward(target_id, learning_rate)


print(f"\n" + "-" * 60)
print(f"Training complete!")
print(f"Loss: {loss_history[0]:.4f} → {loss_history[-1]:.4f}")


# =============================================================================
# PART 5: TEST THE TRAINED MODEL
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: TESTING THE TRAINED MODEL")
print("=" * 70)

print("""
Now our weight matrices contain LEARNED values instead of random values.
Let's see if the model can predict correctly!
""")

test_cases = [
    "the cat sat",      # Should predict "on" (from training)
    "the dog ran",      # Should predict "on" or "fast"
    "the cat",          # Should predict "sat" or "ran"
    "the dog sat on the",  # Should predict "mat" or "floor"
]

print(f"\nPredictions:")
print("-" * 60)

for test_input in test_cases:
    input_ids = [word_to_id[w] for w in test_input.split()]
    probs = model.forward(input_ids)
    
    # Get top 3 predictions
    top3 = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
    
    print(f"\n'{test_input}' → ?")
    for idx, prob in top3:
        print(f"    '{id_to_word[idx]}': {prob*100:.1f}%")


# =============================================================================
# PART 6: COMPARE BEFORE AND AFTER
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: BEFORE vs AFTER TRAINING")
print("=" * 70)

# Create fresh random model for comparison
random_model = TinyLM()

test_input = "the cat sat on the"
input_ids = [word_to_id[w] for w in test_input.split()]

print(f"\nTest: '{test_input}' → ?")
print(f"(Correct answer from training data: 'mat' or 'floor')")

print(f"\nUNTRAINED (random weights):")
random_probs = random_model.forward(input_ids)
for idx, prob in sorted(enumerate(random_probs), key=lambda x: -x[1])[:5]:
    print(f"  '{id_to_word[idx]}': {prob*100:.1f}%")

print(f"\nTRAINED (learned weights):")
trained_probs = model.forward(input_ids)
for idx, prob in sorted(enumerate(trained_probs), key=lambda x: -x[1])[:5]:
    marker = " ✓" if id_to_word[idx] in ["mat", "floor"] else ""
    print(f"  '{id_to_word[idx]}': {prob*100:.1f}%{marker}")


# =============================================================================
# PART 7: VISUALIZE HOW SPECIFIC WEIGHTS CHANGED
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: HOW SPECIFIC WEIGHTS EVOLVED")
print("=" * 70)

print("""
Let's see how individual numbers in the weight matrices changed over training.
Each of these is just one number that got nudged thousands of times!
""")

# ASCII plot of weight evolution
def ascii_line_plot(values, width=50, height=10, title=""):
    """Simple ASCII line plot."""
    if not values:
        return
    
    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val != min_val else 1
    
    # Sample values to fit width
    if len(values) > width:
        indices = np.linspace(0, len(values)-1, width).astype(int)
        sampled = [values[i] for i in indices]
    else:
        sampled = values
        width = len(values)
    
    print(f"\n{title}")
    print("-" * (width + 10))
    
    # Create plot grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot points
    for i, val in enumerate(sampled):
        row = int((max_val - val) / val_range * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][i] = '█'
    
    # Print with y-axis labels
    for row_idx, row in enumerate(grid):
        if row_idx == 0:
            label = f"{max_val:>7.3f}"
        elif row_idx == height - 1:
            label = f"{min_val:>7.3f}"
        else:
            label = "       "
        print(f"{label} │{''.join(row)}│")
    
    print(f"        └{'─' * width}┘")
    print(f"         Epoch 0{' ' * (width - 12)}Epoch {len(values)-1}")

# Plot loss
ascii_line_plot(loss_history, title="LOSS OVER TRAINING (should decrease)")

# Plot specific weights
cat_embed_history = [w['cat_embed_0'] for w in weight_history]
ascii_line_plot(cat_embed_history, title="EMBEDDING FOR 'cat' [dimension 0]")

w_out_history = [w['W_out_0_0'] for w in weight_history]
ascii_line_plot(w_out_history, title="OUTPUT WEIGHT W[0,0]")

print("""
Each wiggle in these plots is the weight being nudged up or down
based on whether that nudge would reduce the prediction error!
""")


# =============================================================================
# PART 8: THE MATH BEHIND ONE WEIGHT UPDATE
# =============================================================================

print("\n" + "=" * 70)
print("PART 8: THE MATH (ONE WEIGHT UPDATE)")
print("=" * 70)

print("""
Let's trace the EXACT math for updating ONE specific weight.

The weight connects hidden neuron 0 to output neuron for "on" (id=6).
""")

# Reset and do one forward pass
model_demo = TinyLM()
input_ids = [word_to_id[w] for w in "the cat sat".split()]
target_id = word_to_id["on"]

# Get the specific weight we'll trace
weight_row = 0  # hidden neuron 0
weight_col = target_id  # output for "on"
old_weight = model_demo.W_output[weight_row, weight_col]

print(f"Weight location: W_output[{weight_row}, {weight_col}]")
print(f"Connects: hidden neuron {weight_row} → output for '{id_to_word[weight_col]}'")
print(f"Current value: {old_weight:.6f}")

# Forward pass
probs = model_demo.forward(input_ids)
hidden_activation = model_demo.hidden[weight_row]  # Value from hidden neuron 0
prob_on = probs[target_id]

print(f"\n1. FORWARD PASS:")
print(f"   Hidden neuron {weight_row} activation: {hidden_activation:.6f}")
print(f"   P('{id_to_word[target_id]}') = {prob_on:.6f}")

# Loss
loss = -np.log(prob_on)
print(f"\n2. LOSS:")
print(f"   Loss = -log({prob_on:.6f}) = {loss:.4f}")

# Gradient calculation
# For softmax + cross-entropy, gradient of loss w.r.t. logit[target] = prob[target] - 1
# For other logits: gradient = prob[that class]
d_logit_on = prob_on - 1  # Gradient for the target class

print(f"\n3. GRADIENT CALCULATION:")
print(f"   d(loss)/d(logit_on) = P('on') - 1 = {prob_on:.6f} - 1 = {d_logit_on:.6f}")
print(f"   (Negative because we want to INCREASE this logit)")

# Gradient for this specific weight uses chain rule:
# d(loss)/d(W[0,6]) = d(loss)/d(logit_on) * d(logit_on)/d(W[0,6])
# logit_on = sum(hidden[i] * W[i, 6]) + bias[6]
# so d(logit_on)/d(W[0,6]) = hidden[0]
d_weight = d_logit_on * hidden_activation

print(f"   d(logit_on)/d(W[{weight_row},{weight_col}]) = hidden[{weight_row}] = {hidden_activation:.6f}")
print(f"   d(loss)/d(W[{weight_row},{weight_col}]) = {d_logit_on:.6f} × {hidden_activation:.6f} = {d_weight:.6f}")

# Update
learning_rate = 0.5
weight_change = -learning_rate * d_weight
new_weight = old_weight + weight_change

print(f"\n4. UPDATE:")
print(f"   learning_rate = {learning_rate}")
print(f"   change = -lr × gradient = -{learning_rate} × {d_weight:.6f} = {weight_change:.6f}")
print(f"   new_weight = {old_weight:.6f} + {weight_change:.6f} = {new_weight:.6f}")

# Do the actual update and verify
model_demo.backward(target_id, learning_rate)
actual_new_weight = model_demo.W_output[weight_row, weight_col]

print(f"\n5. VERIFICATION:")
print(f"   Calculated: {new_weight:.6f}")
print(f"   Actual:     {actual_new_weight:.6f}")
print(f"   ✓ Match!" if abs(new_weight - actual_new_weight) < 1e-6 else "   ✗ Mismatch")

print("""
This is EXACTLY what happens for EVERY weight in EVERY training step.
A 70B model does this for 70,000,000,000 weights per step!
""")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: WHAT 'ADJUSTING WEIGHTS' REALLY MEANS")
print("=" * 70)

print("""
1. WEIGHTS ARE JUST NUMBERS
   - Matrices full of floating-point numbers
   - A 70B model has 70 billion of these numbers
   - They start RANDOM

2. FORWARD PASS USES THE NUMBERS
   - Multiply inputs by weights
   - Add biases
   - Apply activations
   - Get predictions

3. LOSS MEASURES ERROR
   - Compare prediction to correct answer
   - Cross-entropy loss: -log(P_correct)

4. GRADIENTS SHOW DIRECTION
   - For each weight: "should I go up or down?"
   - Calculated via backpropagation (chain rule)

5. UPDATE IS SIMPLE ARITHMETIC
   new_weight = old_weight - learning_rate × gradient
   
   That's it! Each number gets nudged slightly.

6. REPEAT BILLIONS OF TIMES
   - Trillions of tokens
   - Each token = update all weights once
   - Random numbers → learned patterns

THE MAGIC: After billions of tiny nudges, the random numbers
become a next-word predictor that understands language!
""")
