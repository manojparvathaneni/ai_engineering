"""
Neural Networks with PyTorch - Teaching Demo
=============================================

This demo builds from the ground up:
1. Single neuron (y = wx + b)
2. Manual layer implementation
3. PyTorch's Linear layer
4. Practical examples: house prices, spam classification
5. Common layer types and their uses

Run sections incrementally to understand each concept.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("NEURAL NETWORKS WITH PYTORCH - TEACHING DEMO")
print("=" * 60)


# =============================================================================
# PART 1: THE SINGLE NEURON
# =============================================================================

print("\n" + "=" * 60)
print("PART 1: THE SINGLE NEURON")
print("=" * 60)

print("""
The most basic unit: y = wx + b

Where:
  x = input
  w = weight (learned parameter)
  b = bias (learned parameter)
  y = output
""")

# Manual implementation - just Python math
def single_neuron(x, w, b):
    """One neuron: y = wx + b"""
    return w * x + b

# Example: Predicting house price from square footage
square_feet = 1500
weight = 200.0      # $200 per square foot
bias = 50000.0      # base price

price = single_neuron(square_feet, weight, bias)
print(f"House price prediction:")
print(f"  Square feet: {square_feet}")
print(f"  Weight ($/sqft): ${weight}")
print(f"  Bias (base price): ${bias:,.0f}")
print(f"  Predicted price: ${price:,.0f}")


# Same thing with PyTorch tensors
print("\n--- Same calculation with PyTorch tensors ---")
x = torch.tensor([1500.0])
w = torch.tensor([200.0])
b = torch.tensor([50000.0])

y = w * x + b
print(f"  PyTorch result: ${y.item():,.0f}")


# =============================================================================
# PART 2: MULTIPLE INPUTS - VECTOR OPERATIONS
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: MULTIPLE INPUTS")
print("=" * 60)

print("""
With multiple inputs: y = w·x + b (dot product)

Example: House price from [square_feet, bedrooms, bathrooms]
""")

# Multiple features
inputs = torch.tensor([1500.0, 3.0, 2.0])  # sqft, beds, baths
weights = torch.tensor([200.0, 10000.0, 5000.0])  # importance of each
bias = torch.tensor([50000.0])

# Dot product: element-wise multiply, then sum
price = torch.dot(inputs, weights) + bias
print(f"Multi-feature house price prediction:")
print(f"  Inputs: sqft={inputs[0]:.0f}, beds={inputs[1]:.0f}, baths={inputs[2]:.0f}")
print(f"  Weights: {weights.tolist()}")
print(f"  Calculation: (1500×200) + (3×10000) + (2×5000) + 50000")
print(f"  Predicted price: ${price.item():,.0f}")


# =============================================================================
# PART 3: THE LINEAR LAYER (MULTIPLE OUTPUTS)
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: THE LINEAR LAYER")
print("=" * 60)

print("""
A Linear Layer: multiple neurons processing the same input

  Input: [x1, x2, x3]  (3 features)
           ↓
  Output: [y1, y2]     (2 outputs)

Each output has its own set of weights:
  y1 = w11·x1 + w12·x2 + w13·x3 + b1
  y2 = w21·x1 + w22·x2 + w23·x3 + b2

In matrix form: Y = XW^T + B
""")

# Manual implementation
def linear_layer_manual(x, W, b):
    """
    x: input vector (n_features,)
    W: weight matrix (n_outputs, n_features)
    b: bias vector (n_outputs,)
    """
    return torch.matmul(W, x) + b

# 3 inputs → 2 outputs
x = torch.tensor([1.0, 2.0, 3.0])
W = torch.tensor([
    [0.1, 0.2, 0.3],   # weights for output 1
    [0.4, 0.5, 0.6]    # weights for output 2
])
b = torch.tensor([0.1, 0.2])

y = linear_layer_manual(x, W, b)
print(f"Manual linear layer:")
print(f"  Input shape: {x.shape}")
print(f"  Weight shape: {W.shape}")
print(f"  Output: {y}")
print(f"  Output shape: {y.shape}")


# PyTorch's nn.Linear does the same thing
print("\n--- PyTorch nn.Linear ---")

# Create a linear layer: 3 inputs → 2 outputs
linear = nn.Linear(in_features=3, out_features=2)

# Check the shapes PyTorch created
print(f"  Weight shape: {linear.weight.shape}")  # (2, 3)
print(f"  Bias shape: {linear.bias.shape}")      # (2,)

# Forward pass (need batch dimension)
x_batched = torch.tensor([[1.0, 2.0, 3.0]])  # shape: (1, 3)
y = linear(x_batched)
print(f"  Input shape: {x_batched.shape}")
print(f"  Output shape: {y.shape}")
print(f"  Output: {y}")


# =============================================================================
# PART 4: ACTIVATION FUNCTIONS (NON-LINEARITY)
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: ACTIVATION FUNCTIONS")
print("=" * 60)

print("""
Without non-linearity, stacking linear layers is pointless:
  Linear(Linear(x)) = Linear(x)

Activation functions add the non-linearity we need.
""")

# Sample input for demonstrating activations
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"Input: {x.tolist()}")

# ReLU: max(0, x) - most common
relu_out = F.relu(x)
print(f"\nReLU (max(0,x)): {relu_out.tolist()}")
print("  → Negative values become 0, positive pass through")

# Sigmoid: 1/(1+e^-x) - squashes to (0, 1)
sigmoid_out = torch.sigmoid(x)
print(f"\nSigmoid: {[f'{v:.3f}' for v in sigmoid_out.tolist()]}")
print("  → Squashes everything to range (0, 1)")

# Tanh: squashes to (-1, 1)
tanh_out = torch.tanh(x)
print(f"\nTanh: {[f'{v:.3f}' for v in tanh_out.tolist()]}")
print("  → Squashes everything to range (-1, 1)")

# Softmax: converts to probability distribution
logits = torch.tensor([2.0, 1.0, 0.5])
softmax_out = F.softmax(logits, dim=0)
print(f"\nSoftmax on {logits.tolist()}: {[f'{v:.3f}' for v in softmax_out.tolist()]}")
print(f"  → Sum = {softmax_out.sum():.3f} (always sums to 1)")
print("  → Used for multi-class classification")


# =============================================================================
# PART 5: BUILDING A SIMPLE NETWORK
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: BUILDING A SIMPLE NETWORK")
print("=" * 60)

print("""
A neural network = sequence of layers with activations

Let's build: Input(4) → Linear(8) → ReLU → Linear(2) → Output
""")

# Method 1: Using nn.Sequential (simplest)
print("\n--- Method 1: nn.Sequential ---")

simple_net = nn.Sequential(
    nn.Linear(4, 8),    # 4 inputs → 8 hidden
    nn.ReLU(),          # activation
    nn.Linear(8, 2)     # 8 hidden → 2 outputs
)

print(simple_net)

# Forward pass
x = torch.randn(1, 4)  # batch of 1, 4 features
y = simple_net(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {y.shape}")


# Method 2: Custom class (more flexible)
print("\n--- Method 2: Custom nn.Module class ---")

class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

net = SimpleNetwork(input_size=4, hidden_size=8, output_size=2)
print(net)

# Count parameters
total_params = sum(p.numel() for p in net.parameters())
print(f"\nTotal parameters: {total_params}")
print("  Layer 1: 4×8 weights + 8 biases = 40")
print("  Layer 2: 8×2 weights + 2 biases = 18")
print("  Total: 58")


# =============================================================================
# PART 6: PRACTICAL EXAMPLE - HOUSE PRICE REGRESSION
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: HOUSE PRICE PREDICTION (REGRESSION)")
print("=" * 60)

print("""
Task: Predict house price from features
- Input: [square_feet, bedrooms, bathrooms, age]
- Output: price (single number)

This is REGRESSION - predicting a continuous value.
""")

# Create synthetic training data
torch.manual_seed(42)

def generate_house_data(n_samples=100):
    """Generate fake house data with a known formula + noise"""
    sqft = torch.rand(n_samples) * 2000 + 500       # 500-2500 sqft
    beds = torch.randint(1, 6, (n_samples,)).float() # 1-5 bedrooms
    baths = torch.randint(1, 4, (n_samples,)).float() # 1-3 bathrooms
    age = torch.rand(n_samples) * 50                 # 0-50 years old
    
    # True formula (what we want the network to learn)
    price = (sqft * 200) + (beds * 15000) + (baths * 10000) - (age * 1000) + 50000
    price += torch.randn(n_samples) * 10000  # add noise
    
    X = torch.stack([sqft, beds, baths, age], dim=1)
    y = price.unsqueeze(1)
    return X, y

X_train, y_train = generate_house_data(100)
print(f"Training data: {X_train.shape[0]} houses, {X_train.shape[1]} features")
print(f"Price range: ${y_train.min().item():,.0f} - ${y_train.max().item():,.0f}")

# Define the model
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.net(x)

model = HousePriceModel()
print(f"\nModel architecture:")
print(model)

# Training setup
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
print("\nTraining...")
for epoch in range(500):
    # Forward pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        rmse = torch.sqrt(loss).item()
        print(f"  Epoch {epoch+1}: RMSE = ${rmse:,.0f}")

# Test on a new house
print("\n--- Testing on a new house ---")
new_house = torch.tensor([[1800.0, 3.0, 2.0, 10.0]])  # 1800sqft, 3bed, 2bath, 10yrs
predicted_price = model(new_house)
print(f"House: 1800 sqft, 3 bed, 2 bath, 10 years old")
print(f"Predicted price: ${predicted_price.item():,.0f}")

# What the true formula would give
true_price = (1800*200) + (3*15000) + (2*10000) + (-10*1000) + 50000
print(f"True formula price: ${true_price:,.0f}")


# =============================================================================
# PART 7: PRACTICAL EXAMPLE - SPAM CLASSIFICATION
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: SPAM CLASSIFICATION (BINARY CLASSIFICATION)")
print("=" * 60)

print("""
Task: Classify email as spam (1) or not spam (0)
- Input: features extracted from email (word counts, etc.)
- Output: probability of spam

This is CLASSIFICATION - predicting a category.
""")

# Create synthetic email feature data
torch.manual_seed(42)

def generate_email_data(n_samples=200):
    """
    Generate fake email features:
    - feature 0: count of "FREE" (higher in spam)
    - feature 1: count of "URGENT" (higher in spam)
    - feature 2: count of "meeting" (higher in not-spam)
    - feature 3: count of "report" (higher in not-spam)
    - feature 4: number of exclamation marks (higher in spam)
    """
    n_spam = n_samples // 2
    n_ham = n_samples - n_spam
    
    # Spam emails: high FREE, URGENT, !, low meeting, report
    spam_features = torch.randn(n_spam, 5) + torch.tensor([3.0, 2.0, -1.0, -1.0, 4.0])
    spam_labels = torch.ones(n_spam, 1)
    
    # Ham emails: low FREE, URGENT, !, high meeting, report
    ham_features = torch.randn(n_ham, 5) + torch.tensor([-1.0, -1.0, 2.0, 2.0, -1.0])
    ham_labels = torch.zeros(n_ham, 1)
    
    X = torch.cat([spam_features, ham_features], dim=0)
    y = torch.cat([spam_labels, ham_labels], dim=0)
    
    # Shuffle
    perm = torch.randperm(n_samples)
    return X[perm], y[perm]

X_train, y_train = generate_email_data(200)
print(f"Training data: {X_train.shape[0]} emails, {X_train.shape[1]} features")
print(f"Spam ratio: {y_train.mean().item():.1%}")

# Define classifier
class SpamClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )
    
    def forward(self, x):
        return self.net(x)

classifier = SpamClassifier()
print(f"\nModel architecture:")
print(classifier)

# Training
criterion = nn.BCELoss()  # Binary Cross Entropy for classification
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

print("\nTraining...")
for epoch in range(300):
    predictions = classifier(X_train)
    loss = criterion(predictions, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        # Calculate accuracy
        predicted_labels = (predictions > 0.5).float()
        accuracy = (predicted_labels == y_train).float().mean()
        print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.1%}")

# Test on new emails
print("\n--- Testing on new emails ---")

# Spammy email: high FREE, URGENT, !
spammy = torch.tensor([[4.0, 3.0, 0.0, 0.0, 5.0]])
prob = classifier(spammy).item()
print(f"Spammy email (FREE=4, URGENT=3, !=5): {prob:.1%} spam probability")

# Normal email: high meeting, report
normal = torch.tensor([[-1.0, -1.0, 3.0, 3.0, 0.0]])
prob = classifier(normal).item()
print(f"Normal email (meeting=3, report=3): {prob:.1%} spam probability")


# =============================================================================
# PART 8: COMMON LAYER TYPES SHOWCASE
# =============================================================================

print("\n" + "=" * 60)
print("PART 8: COMMON LAYER TYPES")
print("=" * 60)

# --- Embedding Layer ---
print("\n--- Embedding Layer ---")
print("Converts token IDs to dense vectors (used in NLP)")

vocab_size = 1000   # vocabulary has 1000 tokens
embed_dim = 64      # each token becomes a 64-dimensional vector

embedding = nn.Embedding(vocab_size, embed_dim)
print(f"Embedding table shape: {embedding.weight.shape}")

# Look up embeddings for some token IDs
token_ids = torch.tensor([42, 100, 7])  # 3 token IDs
embedded = embedding(token_ids)
print(f"Input token IDs: {token_ids.tolist()}")
print(f"Output shape: {embedded.shape}  (3 tokens × 64 dimensions)")


# --- Dropout Layer ---
print("\n--- Dropout Layer ---")
print("Randomly zeros out neurons during training (prevents overfitting)")

dropout = nn.Dropout(p=0.3)  # 30% dropout rate

x = torch.ones(1, 10)
print(f"Input: {x}")

dropout.train()  # training mode: dropout active
y_train = dropout(x)
print(f"Training output: {y_train}")
print("  → Some values zeroed, others scaled up")

dropout.eval()  # eval mode: dropout disabled
y_eval = dropout(x)
print(f"Eval output: {y_eval}")
print("  → All values pass through unchanged")


# --- Layer Normalization ---
print("\n--- Layer Normalization ---")
print("Normalizes across features (used in transformers)")

layer_norm = nn.LayerNorm(4)  # normalize 4-dimensional vectors

x = torch.tensor([[1.0, 2.0, 3.0, 100.0]])  # note the outlier
print(f"Input: {x}")

y = layer_norm(x)
print(f"Normalized: {y}")
print(f"Mean: {y.mean().item():.4f}, Std: {y.std(unbiased=False).item():.4f}")
print("  → Values normalized to mean≈0, std≈1")


# --- Batch Normalization ---
print("\n--- Batch Normalization ---")
print("Normalizes across batch (used in CNNs)")

batch_norm = nn.BatchNorm1d(3)  # 3 features

# Batch of 4 samples, each with 3 features
x = torch.tensor([
    [1.0, 100.0, 0.5],
    [2.0, 200.0, 0.6],
    [3.0, 300.0, 0.7],
    [4.0, 400.0, 0.8]
])
print(f"Input shape: {x.shape}")
print(f"Feature means before: {x.mean(dim=0)}")

batch_norm.train()
y = batch_norm(x)
print(f"Feature means after: {y.mean(dim=0)}")
print("  → Each feature normalized across the batch")


# =============================================================================
# PART 9: BUILDING TOWARD TRANSFORMERS
# =============================================================================

print("\n" + "=" * 60)
print("PART 9: BUILDING TOWARD TRANSFORMERS")
print("=" * 60)

print("""
A Transformer block combines several layer types:

Input (tokens as vectors)
    ↓
[Self-Attention] → captures relationships between tokens
    ↓
Add & LayerNorm → residual connection + normalization
    ↓
[Feed-Forward] → two linear layers with activation
    ↓
Add & LayerNorm → residual connection + normalization
    ↓
Output (same shape, enriched with context)
""")

class SimplifiedTransformerBlock(nn.Module):
    """
    Simplified transformer block (attention details coming next!)
    This shows the overall structure without the attention math.
    """
    def __init__(self, d_model=64):
        super().__init__()
        
        # Placeholder for attention (we'll build this next)
        self.attention_placeholder = nn.Linear(d_model, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network (two linear layers)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # expand
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)   # contract back
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        attended = self.attention_placeholder(x)
        x = self.norm1(x + attended)  # add & norm
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)    # add & norm
        
        return x

# Demo
block = SimplifiedTransformerBlock(d_model=64)
print("Simplified Transformer Block:")
print(block)

# Input: batch of 2, sequence of 10 tokens, each 64-dimensional
x = torch.randn(2, 10, 64)
y = block(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {y.shape}")
print("  → Same shape! Transformers preserve sequence dimensions.")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("""
What we've covered:

1. SINGLE NEURON: y = wx + b
   - The atom that everything is built from

2. LINEAR LAYER: Y = WX + B
   - Multiple neurons in parallel
   - Matrix multiplication

3. ACTIVATION FUNCTIONS: ReLU, Sigmoid, Tanh, Softmax
   - Add non-linearity so networks can learn complex patterns

4. BUILDING NETWORKS:
   - nn.Sequential for simple architectures
   - nn.Module for custom architectures

5. PRACTICAL TASKS:
   - Regression: predict continuous values (house prices)
   - Classification: predict categories (spam detection)

6. LAYER TYPES:
   - Linear: general transformation
   - Embedding: token IDs → vectors
   - Dropout: regularization
   - LayerNorm/BatchNorm: stabilize training

7. TOWARD TRANSFORMERS:
   - Attention + Feed-Forward + Residuals + Normalization
   - Next: dive into how attention actually works!

Key insight: Neural networks are just math!
- Matrix multiplications
- Simple non-linear functions
- Chained together
- With millions of learned parameters
""")
