"""
Attention Mechanism Deep Dive - Teaching Demo
==============================================

This demo builds attention from scratch:
1. Q, K, V matrix operations (pure numpy)
2. Scaled dot-product attention step-by-step
3. Visualize attention weights
4. Multi-head attention
5. Causal masking for decoder-only models
6. Full self-attention layer

Run this to see exactly how attention works!
"""

import numpy as np
np.set_printoptions(precision=3, suppress=True)

print("=" * 70)
print("ATTENTION MECHANISM DEEP DIVE")
print("=" * 70)


# =============================================================================
# PART 1: THE BUILDING BLOCKS - Q, K, V
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: QUERY, KEY, VALUE PROJECTIONS")
print("=" * 70)

print("""
Every word embedding gets projected into three different representations:
  - Query (Q): "What am I looking for?"
  - Key (K): "What do I have to offer?"
  - Value (V): "What information do I carry?"

The projections are learned linear transformations:
  Q = X × Wq
  K = X × Wk
  V = X × Wv
""")

# Simple example: 3 words, 4-dimensional embeddings
np.random.seed(42)

# Word embeddings (imagine these came from an embedding layer)
# Shape: (seq_len=3, d_model=4)
words = ["The", "cat", "sat"]
X = np.array([
    [0.2, 0.5, 0.1, 0.8],   # "The"
    [0.9, 0.1, 0.4, 0.3],   # "cat"
    [0.3, 0.7, 0.2, 0.6],   # "sat"
])

print(f"Input embeddings X (shape {X.shape}):")
for i, word in enumerate(words):
    print(f"  '{word}': {X[i]}")

# Learned projection matrices (in practice, these are trained)
d_model = 4
d_k = 4  # dimension of Q, K, V (often same as d_model for single-head)

Wq = np.random.randn(d_model, d_k) * 0.5
Wk = np.random.randn(d_model, d_k) * 0.5
Wv = np.random.randn(d_model, d_k) * 0.5

print(f"\nWeight matrices (each {d_model}×{d_k}):")
print(f"  Wq parameters: {Wq.size}")
print(f"  Wk parameters: {Wk.size}")
print(f"  Wv parameters: {Wv.size}")
print(f"  Total attention parameters: {3 * Wq.size}")

# Project to Q, K, V
Q = X @ Wq  # (3, 4) @ (4, 4) = (3, 4)
K = X @ Wk
V = X @ Wv

print(f"\nQuery matrix Q (shape {Q.shape}):")
for i, word in enumerate(words):
    print(f"  Q['{word}']: {Q[i]}")

print(f"\nKey matrix K (shape {K.shape}):")
for i, word in enumerate(words):
    print(f"  K['{word}']: {K[i]}")

print(f"\nValue matrix V (shape {V.shape}):")
for i, word in enumerate(words):
    print(f"  V['{word}']: {V[i]}")


# =============================================================================
# PART 2: SCALED DOT-PRODUCT ATTENTION (STEP BY STEP)
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: SCALED DOT-PRODUCT ATTENTION")
print("=" * 70)

print("""
The attention formula:
    Attention(Q, K, V) = softmax(Q × Kᵀ / √d_k) × V

Let's break it down step by step...
""")

# Step 1: Q × Kᵀ (attention scores)
print("STEP 1: Compute attention scores (Q × Kᵀ)")
print("-" * 50)

scores = Q @ K.T  # (3, 4) @ (4, 3) = (3, 3)

print(f"Scores matrix (shape {scores.shape}):")
print(f"  scores[i,j] = how much word i should attend to word j\n")

# Pretty print with labels
print("         ", end="")
for w in words:
    print(f"{w:>8}", end="")
print()
for i, w in enumerate(words):
    print(f"  {w:>5}  ", end="")
    for j in range(len(words)):
        print(f"{scores[i,j]:>8.3f}", end="")
    print()

print(f"\nInterpretation:")
print(f"  'cat' attending to 'cat' has score {scores[1,1]:.3f} (self-attention)")
print(f"  'sat' attending to 'cat' has score {scores[2,1]:.3f}")


# Step 2: Scale by √d_k
print("\n\nSTEP 2: Scale by √d_k")
print("-" * 50)

d_k = K.shape[-1]
scale = np.sqrt(d_k)
scaled_scores = scores / scale

print(f"d_k = {d_k}, √d_k = {scale:.2f}")
print(f"\nWhy scale?")
print(f"  - Without scaling, large d_k → large dot products")
print(f"  - Large values → extreme softmax (one word gets ~100%)")
print(f"  - Extreme softmax → vanishing gradients")
print(f"\nScaled scores:")

print("         ", end="")
for w in words:
    print(f"{w:>8}", end="")
print()
for i, w in enumerate(words):
    print(f"  {w:>5}  ", end="")
    for j in range(len(words)):
        print(f"{scaled_scores[i,j]:>8.3f}", end="")
    print()


# Step 3: Softmax
print("\n\nSTEP 3: Apply softmax (convert to probabilities)")
print("-" * 50)

def softmax(x, axis=-1):
    """Compute softmax along specified axis."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

attention_weights = softmax(scaled_scores, axis=-1)

print("Attention weights (each row sums to 1.0):")
print("         ", end="")
for w in words:
    print(f"{w:>8}", end="")
print("    sum")
for i, w in enumerate(words):
    print(f"  {w:>5}  ", end="")
    for j in range(len(words)):
        print(f"{attention_weights[i,j]:>8.3f}", end="")
    print(f"  = {attention_weights[i].sum():.3f}")

print(f"\nInterpretation:")
print(f"  'cat' pays {attention_weights[1,1]*100:.1f}% attention to itself")
print(f"  'cat' pays {attention_weights[1,0]*100:.1f}% attention to 'The'")
print(f"  'cat' pays {attention_weights[1,2]*100:.1f}% attention to 'sat'")


# Step 4: Weighted sum of Values
print("\n\nSTEP 4: Compute weighted sum of Values")
print("-" * 50)

output = attention_weights @ V  # (3, 3) @ (3, 4) = (3, 4)

print(f"Output = Attention_weights × V")
print(f"Shape: ({attention_weights.shape[0]}, {attention_weights.shape[1]}) @ ({V.shape[0]}, {V.shape[1]}) = {output.shape}")

print(f"\nOutput embeddings (now context-aware!):")
for i, word in enumerate(words):
    print(f"  '{word}': {output[i]}")

print(f"\nThe output for 'cat' is a weighted combination of:")
print(f"  {attention_weights[1,0]:.3f} × V['The'] +")
print(f"  {attention_weights[1,1]:.3f} × V['cat'] +")
print(f"  {attention_weights[1,2]:.3f} × V['sat']")


# =============================================================================
# PART 3: VISUALIZING ATTENTION WEIGHTS
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: VISUALIZING ATTENTION")
print("=" * 70)

def visualize_attention(weights, labels, title="Attention Weights"):
    """Create ASCII visualization of attention weights."""
    print(f"\n{title}")
    print("=" * 50)
    
    n = len(labels)
    max_label = max(len(l) for l in labels)
    
    # Header
    print(" " * (max_label + 3), end="")
    for l in labels:
        print(f"{l:>8}", end="")
    print()
    
    # Rows
    for i, label in enumerate(labels):
        print(f"{label:>{max_label}}  │", end="")
        for j in range(n):
            # Use ASCII art to show weight magnitude
            w = weights[i, j]
            if w > 0.5:
                char = "███"
            elif w > 0.3:
                char = "▓▓▓"
            elif w > 0.15:
                char = "▒▒▒"
            elif w > 0.05:
                char = "░░░"
            else:
                char = "   "
            print(f"  {char}  ", end="")
        print()
    
    print("\nLegend: ███ >50%  ▓▓▓ >30%  ▒▒▒ >15%  ░░░ >5%")

visualize_attention(attention_weights, words, "Attention Pattern")

# Let's try a more interesting example
print("\n\nMore interesting example: 'The cat sat on it'")
print("-" * 50)

np.random.seed(123)
words2 = ["The", "cat", "sat", "on", "it"]
seq_len = len(words2)
d = 8

# Random embeddings and projections
X2 = np.random.randn(seq_len, d)
Wq2 = np.random.randn(d, d) * 0.3
Wk2 = np.random.randn(d, d) * 0.3
Wv2 = np.random.randn(d, d) * 0.3

Q2 = X2 @ Wq2
K2 = X2 @ Wk2
V2 = X2 @ Wv2

scores2 = Q2 @ K2.T / np.sqrt(d)
weights2 = softmax(scores2, axis=-1)

visualize_attention(weights2, words2, "Attention: 'The cat sat on it'")

print("\nNote: This is random initialization - trained weights would show")
print("'it' strongly attending to 'cat' (the referent)!")


# =============================================================================
# PART 4: CAUSAL MASKING (DECODER-ONLY)
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: CAUSAL MASKING")
print("=" * 70)

print("""
For language generation, words can only attend to PREVIOUS words.
This is enforced with a causal mask.

Before softmax, we set future positions to -infinity:
  -∞ → softmax → 0 attention
""")

def create_causal_mask(seq_len):
    """Create lower triangular mask (True = attend, False = mask)."""
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask

def apply_causal_mask(scores, mask):
    """Apply causal mask by setting masked positions to -inf."""
    masked_scores = scores.copy()
    masked_scores[mask == 0] = -np.inf
    return masked_scores

# Create and show the mask
mask = create_causal_mask(5)
print("Causal mask (1 = can attend, 0 = blocked):")
print("         ", end="")
for w in words2:
    print(f"{w:>8}", end="")
print()
for i, w in enumerate(words2):
    print(f"  {w:>5}  ", end="")
    for j in range(len(words2)):
        print(f"{int(mask[i,j]):>8}", end="")
    print()

# Apply mask to our attention scores
print("\n\nScores BEFORE masking:")
scores_unmasked = Q2 @ K2.T / np.sqrt(d)
print("         ", end="")
for w in words2:
    print(f"{w:>8}", end="")
print()
for i, w in enumerate(words2):
    print(f"  {w:>5}  ", end="")
    for j in range(len(words2)):
        print(f"{scores_unmasked[i,j]:>8.2f}", end="")
    print()

print("\n\nScores AFTER masking (future = -inf):")
scores_masked = apply_causal_mask(scores_unmasked, mask)
print("         ", end="")
for w in words2:
    print(f"{w:>8}", end="")
print()
for i, w in enumerate(words2):
    print(f"  {w:>5}  ", end="")
    for j in range(len(words2)):
        if scores_masked[i,j] == -np.inf:
            print(f"    -inf", end="")
        else:
            print(f"{scores_masked[i,j]:>8.2f}", end="")
    print()

# Attention weights after masking
weights_causal = softmax(scores_masked, axis=-1)
print("\n\nCausal attention weights (after softmax):")
print("         ", end="")
for w in words2:
    print(f"{w:>8}", end="")
print()
for i, w in enumerate(words2):
    print(f"  {w:>5}  ", end="")
    for j in range(len(words2)):
        print(f"{weights_causal[i,j]:>8.3f}", end="")
    print()

print("\nNotice:")
print("  - 'The' only attends to itself (first word)")
print("  - 'it' can attend to all previous words")
print("  - No word attends to future words (upper triangle is 0)")

visualize_attention(weights_causal, words2, "Causal Attention Pattern")


# =============================================================================
# PART 5: MULTI-HEAD ATTENTION
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: MULTI-HEAD ATTENTION")
print("=" * 70)

print("""
Instead of one attention pattern, compute multiple in parallel.
Each "head" can learn different relationships:
  - Head 1: Maybe syntactic (subject-verb)
  - Head 2: Maybe semantic (adjective-noun)
  - Head 3: Maybe positional (nearby words)
  
Process:
  1. Split d_model into h heads of size d_k = d_model/h
  2. Each head does independent attention
  3. Concatenate results
  4. Project back to d_model
""")

def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, mask=None):
    """
    Multi-head attention from scratch.
    
    Args:
        X: Input (seq_len, d_model)
        Wq, Wk, Wv: Projection weights (d_model, d_model)
        Wo: Output projection (d_model, d_model)
        n_heads: Number of attention heads
        mask: Optional causal mask
    """
    seq_len, d_model = X.shape
    d_k = d_model // n_heads
    
    # Project to Q, K, V
    Q = X @ Wq  # (seq_len, d_model)
    K = X @ Wk
    V = X @ Wv
    
    # Reshape for multi-head: (seq_len, n_heads, d_k)
    Q = Q.reshape(seq_len, n_heads, d_k)
    K = K.reshape(seq_len, n_heads, d_k)
    V = V.reshape(seq_len, n_heads, d_k)
    
    # Transpose to (n_heads, seq_len, d_k) for batched attention
    Q = Q.transpose(1, 0, 2)
    K = K.transpose(1, 0, 2)
    V = V.transpose(1, 0, 2)
    
    # Compute attention for each head
    # scores: (n_heads, seq_len, seq_len)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)
    
    weights = softmax(scores, axis=-1)
    
    # Weighted sum of values
    # (n_heads, seq_len, seq_len) @ (n_heads, seq_len, d_k) = (n_heads, seq_len, d_k)
    attended = np.matmul(weights, V)
    
    # Transpose back and concatenate heads
    # (n_heads, seq_len, d_k) → (seq_len, n_heads, d_k) → (seq_len, d_model)
    attended = attended.transpose(1, 0, 2).reshape(seq_len, d_model)
    
    # Final output projection
    output = attended @ Wo
    
    return output, weights

# Demo multi-head attention
print("\nExample: 4 heads, d_model=16")
print("-" * 50)

np.random.seed(42)
d_model = 16
n_heads = 4
d_k = d_model // n_heads  # 4

words3 = ["The", "cat", "sat"]
seq_len = len(words3)

X3 = np.random.randn(seq_len, d_model)
Wq3 = np.random.randn(d_model, d_model) * 0.3
Wk3 = np.random.randn(d_model, d_model) * 0.3
Wv3 = np.random.randn(d_model, d_model) * 0.3
Wo3 = np.random.randn(d_model, d_model) * 0.3

mask3 = create_causal_mask(seq_len)

output3, head_weights = multi_head_attention(X3, Wq3, Wk3, Wv3, Wo3, n_heads, mask3)

print(f"Input shape: {X3.shape}")
print(f"Output shape: {output3.shape}")
print(f"Number of heads: {n_heads}")
print(f"Per-head dimension (d_k): {d_k}")

# Show attention pattern for each head
print(f"\nAttention patterns by head:")
for h in range(n_heads):
    print(f"\n  Head {h+1}:")
    print("           ", end="")
    for w in words3:
        print(f"{w:>8}", end="")
    print()
    for i, w in enumerate(words3):
        print(f"    {w:>5}  ", end="")
        for j in range(len(words3)):
            print(f"{head_weights[h, i, j]:>8.3f}", end="")
        print()

print("\nEach head learns different patterns!")
print("(In trained models, heads specialize in different relationships)")


# =============================================================================
# PART 6: COMPLETE SELF-ATTENTION LAYER
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: COMPLETE SELF-ATTENTION LAYER")
print("=" * 70)

class SelfAttention:
    """
    Complete self-attention layer implementation.
    Matches the structure used in transformers like GPT.
    """
    
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Initialize weights (in practice, use proper initialization)
        scale = 0.02
        self.Wq = np.random.randn(d_model, d_model) * scale
        self.Wk = np.random.randn(d_model, d_model) * scale
        self.Wv = np.random.randn(d_model, d_model) * scale
        self.Wo = np.random.randn(d_model, d_model) * scale
        
        # Count parameters
        self.n_params = 4 * d_model * d_model
    
    def __call__(self, X, causal=True):
        """
        Forward pass.
        
        Args:
            X: Input tensor (seq_len, d_model)
            causal: Whether to use causal masking
            
        Returns:
            output: Attended output (seq_len, d_model)
            weights: Attention weights (n_heads, seq_len, seq_len)
        """
        seq_len = X.shape[0]
        
        # Create causal mask if needed
        mask = create_causal_mask(seq_len) if causal else None
        
        # Multi-head attention
        output, weights = multi_head_attention(
            X, self.Wq, self.Wk, self.Wv, self.Wo, 
            self.n_heads, mask
        )
        
        return output, weights
    
    def __repr__(self):
        return f"SelfAttention(d_model={self.d_model}, n_heads={self.n_heads}, params={self.n_params:,})"

# Create and use a self-attention layer
print("\nCreating self-attention layer:")
print("-" * 50)

attn = SelfAttention(d_model=64, n_heads=8)
print(attn)

# Test with sample input
test_input = np.random.randn(10, 64)  # 10 tokens, 64 dimensions
test_output, test_weights = attn(test_input, causal=True)

print(f"\nInput shape:  {test_input.shape}")
print(f"Output shape: {test_output.shape}")
print(f"Weights shape: {test_weights.shape} (n_heads, seq_len, seq_len)")

print("\n✓ Self-attention preserves sequence length and dimension!")
print("✓ Output is now context-aware (each token sees other tokens)")


# =============================================================================
# PART 7: COMPARISON WITH PYTORCH
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: PYTORCH EQUIVALENT (for reference)")
print("=" * 70)

pytorch_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """PyTorch self-attention matching our numpy implementation."""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, causal=True):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.Wq(x)  # (batch, seq, d_model)
        K = self.Wk(x)
        V = self.Wv(x)
        
        # Reshape for multi-head attention
        # (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Causal mask
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax and weighted sum
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.Wo(attended), weights


# Usage
attn = SelfAttention(d_model=64, n_heads=8)
x = torch.randn(2, 10, 64)  # batch=2, seq_len=10, d_model=64
output, weights = attn(x)

print(f"Input: {x.shape}")
print(f"Output: {output.shape}")
print(f"Weights: {weights.shape}")

# Count parameters
params = sum(p.numel() for p in attn.parameters())
print(f"Parameters: {params:,}")  # 4 * 64 * 64 = 16,384
'''

print(pytorch_code)


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
ATTENTION MECHANISM - KEY CONCEPTS

1. Q, K, V PROJECTIONS
   Q = X × Wq  (what am I looking for?)
   K = X × Wk  (what do I offer?)
   V = X × Wv  (what info do I carry?)

2. SCALED DOT-PRODUCT ATTENTION
   Attention(Q,K,V) = softmax(QKᵀ/√d_k) × V
   
   Step by step:
   - QKᵀ: Compare all queries to all keys (scores)
   - /√d_k: Scale to prevent extreme softmax
   - softmax: Convert to probabilities (rows sum to 1)
   - × V: Weighted combination of values

3. CAUSAL MASKING (for decoders)
   - Set future positions to -∞ before softmax
   - Ensures each position only sees past positions
   - Creates lower triangular attention pattern

4. MULTI-HEAD ATTENTION
   - Multiple attention patterns in parallel
   - Each head can specialize in different relationships
   - Concat heads → project back to d_model

5. PARAMETER COUNT
   - Single head: 4 × d_model² (Q, K, V, Output projections)
   - Multi-head: Same total, just organized differently

6. KEY INSIGHT
   Attention lets each token dynamically decide which other tokens
   are relevant, enabling long-range dependencies and context-aware
   representations.
""")


# Quick reference
print("\n" + "-" * 70)
print("QUICK REFERENCE FORMULAS")
print("-" * 70)
print("""
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Q = X × Wq    (seq × d_model) × (d_model × d_k) = (seq × d_k)
K = X × Wk
V = X × Wv

Scores = Q × K^T    (seq × d_k) × (d_k × seq) = (seq × seq)
Weights = softmax(Scores / √d_k)    (seq × seq), rows sum to 1
Output = Weights × V    (seq × seq) × (seq × d_v) = (seq × d_v)

Parameters per attention layer: 4 × d_model²
""")
