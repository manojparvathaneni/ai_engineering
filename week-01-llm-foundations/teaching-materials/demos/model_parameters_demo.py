"""
Model Parameters Deep Dive - Teaching Demo
==========================================

This demo helps you understand:
1. What parameters are and where they live
2. How to calculate parameter counts
3. Memory requirements for different precisions
4. Reading and interpreting model specifications

Works with pure Python (no PyTorch required for calculations).
PyTorch examples included for when you have it available.
"""

print("=" * 70)
print("MODEL PARAMETERS DEEP DIVE")
print("=" * 70)


# =============================================================================
# PART 1: PARAMETER COUNTING BASICS
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: PARAMETER COUNTING BASICS")
print("=" * 70)

print("""
Every neural network layer has learnable parameters:
  - Weights (W): The transformation matrix
  - Biases (b): The offset values

Linear layer formula: y = Wx + b
Parameters = weights + biases = (in × out) + out
""")

def count_linear_params(in_features, out_features, bias=True):
    """Count parameters in a linear layer."""
    weights = in_features * out_features
    biases = out_features if bias else 0
    return weights + biases

# Examples
print("Linear Layer Parameter Counts:")
print("-" * 50)

examples = [
    (768, 768, "Small attention projection"),
    (768, 3072, "Small MLP up-projection"),
    (3072, 768, "Small MLP down-projection"),
    (4096, 4096, "Llama 7B attention projection"),
    (4096, 11008, "Llama 7B MLP up-projection"),
    (12288, 12288, "GPT-3 175B attention projection"),
]

for in_f, out_f, name in examples:
    params = count_linear_params(in_f, out_f)
    print(f"  {name}")
    print(f"    {in_f:,} → {out_f:,} = {params:,} params ({params/1e6:.2f}M)")


# =============================================================================
# PART 2: TRANSFORMER BLOCK PARAMETERS
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: TRANSFORMER BLOCK PARAMETERS")
print("=" * 70)

print("""
A transformer block has:
  1. Self-Attention (Q, K, V projections + output projection)
  2. MLP / Feed-Forward (up projection + down projection)
  3. Layer Norms (2 per block)
""")

def count_attention_params(d_model, n_heads, n_kv_heads=None):
    """
    Count parameters in attention layer.
    
    Standard attention: Q, K, V each have d_model × d_model
    Grouped-Query Attention (GQA): K, V are smaller
    """
    if n_kv_heads is None:
        n_kv_heads = n_heads  # Standard multi-head attention
    
    d_head = d_model // n_heads
    
    # Q projection: d_model → d_model (or n_heads × d_head)
    q_params = d_model * d_model + d_model
    
    # K, V projections: may be smaller with GQA
    kv_dim = n_kv_heads * d_head
    k_params = d_model * kv_dim + kv_dim
    v_params = d_model * kv_dim + kv_dim
    
    # Output projection: d_model → d_model
    o_params = d_model * d_model + d_model
    
    return {
        'q': q_params,
        'k': k_params,
        'v': v_params,
        'o': o_params,
        'total': q_params + k_params + v_params + o_params
    }

def count_mlp_params(d_model, d_ff):
    """Count parameters in MLP (feed-forward) layer."""
    # Up projection: d_model → d_ff
    up_params = d_model * d_ff + d_ff
    # Down projection: d_ff → d_model
    down_params = d_ff * d_model + d_model
    
    return {
        'up': up_params,
        'down': down_params,
        'total': up_params + down_params
    }

def count_layernorm_params(d_model):
    """Count parameters in layer norm (scale and shift)."""
    return 2 * d_model  # gamma and beta

def count_block_params(d_model, n_heads, d_ff, n_kv_heads=None):
    """Count total parameters in one transformer block."""
    attn = count_attention_params(d_model, n_heads, n_kv_heads)
    mlp = count_mlp_params(d_model, d_ff)
    ln = 2 * count_layernorm_params(d_model)  # 2 layer norms per block
    
    return {
        'attention': attn['total'],
        'mlp': mlp['total'],
        'layernorm': ln,
        'total': attn['total'] + mlp['total'] + ln
    }

# Example: GPT-2 Small block
print("\nGPT-2 Small (one block):")
print("-" * 50)
block = count_block_params(d_model=768, n_heads=12, d_ff=3072)
print(f"  Attention: {block['attention']:,} ({block['attention']/1e6:.2f}M)")
print(f"  MLP:       {block['mlp']:,} ({block['mlp']/1e6:.2f}M)")
print(f"  LayerNorm: {block['layernorm']:,}")
print(f"  Total:     {block['total']:,} ({block['total']/1e6:.2f}M)")


# =============================================================================
# PART 3: FULL MODEL PARAMETER CALCULATION
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: FULL MODEL PARAMETERS")
print("=" * 70)

def count_model_params(vocab_size, d_model, n_layers, n_heads, d_ff, 
                       n_kv_heads=None, tie_embeddings=True):
    """
    Count total parameters in a decoder-only transformer.
    
    Args:
        vocab_size: Size of token vocabulary
        d_model: Hidden dimension
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads
        d_ff: Feed-forward intermediate dimension
        n_kv_heads: Number of key/value heads (for GQA)
        tie_embeddings: Whether input/output embeddings are shared
    """
    # Token embedding
    embedding = vocab_size * d_model
    
    # Transformer blocks
    block = count_block_params(d_model, n_heads, d_ff, n_kv_heads)
    all_blocks = n_layers * block['total']
    
    # Final layer norm
    final_ln = count_layernorm_params(d_model)
    
    # Output projection (LM head)
    if tie_embeddings:
        lm_head = 0  # Shared with embedding
    else:
        lm_head = d_model * vocab_size
    
    total = embedding + all_blocks + final_ln + lm_head
    
    return {
        'embedding': embedding,
        'per_block': block['total'],
        'all_blocks': all_blocks,
        'final_ln': final_ln,
        'lm_head': lm_head,
        'total': total
    }

def format_params(n):
    """Format parameter count nicely."""
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    elif n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    else:
        return f"{n:,}"


# GPT-2 Family
print("\nGPT-2 Family:")
print("-" * 70)

gpt2_configs = [
    ("GPT-2 Small",  50257, 768,  12, 12, 3072),
    ("GPT-2 Medium", 50257, 1024, 24, 16, 4096),
    ("GPT-2 Large",  50257, 1280, 36, 20, 5120),
    ("GPT-2 XL",     50257, 1600, 48, 25, 6400),
]

print(f"{'Model':<15} {'d_model':>8} {'layers':>7} {'heads':>6} {'Calculated':>12} {'Official':>10}")
print("-" * 70)
for name, vocab, d_model, layers, heads, d_ff in gpt2_configs:
    params = count_model_params(vocab, d_model, layers, heads, d_ff)
    official = {"GPT-2 Small": "117M", "GPT-2 Medium": "345M", 
                "GPT-2 Large": "762M", "GPT-2 XL": "1.5B"}
    print(f"{name:<15} {d_model:>8} {layers:>7} {heads:>6} {format_params(params['total']):>12} {official[name]:>10}")


# Llama Family
print("\n\nLlama 2 Family:")
print("-" * 70)

llama_configs = [
    ("Llama 2 7B",  32000, 4096,  32, 32, 11008, 32),    # n_kv_heads = n_heads (no GQA)
    ("Llama 2 13B", 32000, 5120,  40, 40, 13824, 40),
    ("Llama 2 70B", 32000, 8192,  80, 64, 28672, 8),     # GQA: 64 heads, 8 KV heads
]

print(f"{'Model':<15} {'d_model':>8} {'layers':>7} {'heads':>6} {'kv_heads':>9} {'Calculated':>12}")
print("-" * 70)
for name, vocab, d_model, layers, heads, d_ff, kv_heads in llama_configs:
    params = count_model_params(vocab, d_model, layers, heads, d_ff, 
                                n_kv_heads=kv_heads, tie_embeddings=False)
    print(f"{name:<15} {d_model:>8} {layers:>7} {heads:>6} {kv_heads:>9} {format_params(params['total']):>12}")


# GPT-3 175B
print("\n\nGPT-3 175B Breakdown:")
print("-" * 70)

gpt3_params = count_model_params(
    vocab_size=50257,
    d_model=12288,
    n_layers=96,
    n_heads=96,
    d_ff=49152  # 4 × d_model
)

print(f"  Token Embedding:  {format_params(gpt3_params['embedding']):>15}")
print(f"  Per Block:        {format_params(gpt3_params['per_block']):>15}")
print(f"  All Blocks (96):  {format_params(gpt3_params['all_blocks']):>15}")
print(f"  Final LayerNorm:  {format_params(gpt3_params['final_ln']):>15}")
print(f"  LM Head:          {format_params(gpt3_params['lm_head']):>15}")
print(f"  {'─' * 30}")
print(f"  TOTAL:            {format_params(gpt3_params['total']):>15}")
print(f"\n  Official: 175B parameters ✓")


# =============================================================================
# PART 4: MEMORY REQUIREMENTS
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: MEMORY REQUIREMENTS")
print("=" * 70)

print("""
Parameters need memory! Different precisions = different memory:
  FP32 (float32): 4 bytes per parameter
  FP16 (float16): 2 bytes per parameter
  BF16 (bfloat16): 2 bytes per parameter
  INT8: 1 byte per parameter
  INT4: 0.5 bytes per parameter
""")

def calculate_memory(n_params, precision='fp16'):
    """Calculate memory in GB for given parameters and precision."""
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5
    }
    bytes_total = n_params * bytes_per_param[precision]
    gb = bytes_total / (1024 ** 3)
    return gb

models = [
    ("GPT-2 Small", 117e6),
    ("Llama 2 7B", 7e9),
    ("Llama 2 13B", 13e9),
    ("Llama 2 70B", 70e9),
    ("Llama 3 405B", 405e9),
    ("GPT-3 175B", 175e9),
]

print("\nModel Memory Requirements (Inference):")
print("-" * 80)
print(f"{'Model':<15} {'Params':>10} {'FP32':>10} {'FP16':>10} {'INT8':>10} {'INT4':>10}")
print("-" * 80)

for name, params in models:
    fp32 = calculate_memory(params, 'fp32')
    fp16 = calculate_memory(params, 'fp16')
    int8 = calculate_memory(params, 'int8')
    int4 = calculate_memory(params, 'int4')
    print(f"{name:<15} {format_params(params):>10} {fp32:>9.1f}GB {fp16:>9.1f}GB {int8:>9.1f}GB {int4:>9.1f}GB")

print("\nTraining Memory (rough estimates, includes gradients + optimizer):")
print("-" * 60)
print("  Training typically requires 4-20× inference memory")
print("  With Adam optimizer (FP32 training): ~16× model size")
print("  With mixed precision + gradient checkpointing: ~4-6× model size")


# =============================================================================
# PART 5: THE 12 × N × D² APPROXIMATION
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: THE 12 × N × D² APPROXIMATION")
print("=" * 70)

print("""
Quick estimate for transformer parameters:

    Parameters ≈ 12 × n_layers × d_model²

Why 12?
  - Attention: Q, K, V, O projections = 4 × d_model²
  - MLP: up + down projections = 2 × d_ff = 2 × 4 × d_model² = 8 × d_model²
  - Total per block ≈ 12 × d_model²

(This ignores embeddings, biases, layer norms - good for rough estimates)
""")

def quick_estimate(n_layers, d_model):
    """Quick parameter estimate using 12 × N × D² rule."""
    return 12 * n_layers * (d_model ** 2)

print("Quick Estimates vs Actual:")
print("-" * 60)

test_models = [
    ("GPT-2 Small", 12, 768, 117e6),
    ("GPT-2 XL", 48, 1600, 1.5e9),
    ("Llama 7B", 32, 4096, 7e9),
    ("GPT-3 175B", 96, 12288, 175e9),
]

print(f"{'Model':<15} {'n_layers':>9} {'d_model':>8} {'Estimate':>12} {'Actual':>12} {'Error':>8}")
print("-" * 70)
for name, layers, d_model, actual in test_models:
    estimate = quick_estimate(layers, d_model)
    error = abs(estimate - actual) / actual * 100
    print(f"{name:<15} {layers:>9} {d_model:>8} {format_params(estimate):>12} {format_params(actual):>12} {error:>7.1f}%")


# =============================================================================
# PART 6: UNDERSTANDING MODEL CARDS
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: READING MODEL CARDS")
print("=" * 70)

model_card = """
Example Model Card: Llama 3 70B
================================

Architecture:        Decoder-only Transformer
Parameters:          70.6B
Hidden size:         8192
Intermediate size:   28672
Number of layers:    80
Attention heads:     64
KV heads:            8     ← Grouped-Query Attention!
Head dimension:      128
Vocabulary size:     128,256
Max sequence length: 8,192
Training tokens:     15T
"""
print(model_card)

print("\nWhat each spec tells you:")
print("-" * 60)
specs = [
    ("70.6B parameters", "Large model, needs ~35GB+ VRAM (INT4)"),
    ("d_model = 8192", "Rich 8K-dimensional token representations"),
    ("80 layers", "Very deep - good complex reasoning"),
    ("64 attn heads, 8 KV heads", "GQA: 8× memory savings in KV cache"),
    ("128K vocabulary", "Large vocab - multilingual, code-friendly"),
    ("8K context", "Can process ~6000 words at once"),
    ("15T training tokens", "Massive training (Chinchilla: 1.4T optimal)"),
]

for spec, meaning in specs:
    print(f"  {spec:<30} → {meaning}")


# =============================================================================
# PART 7: GROUPED-QUERY ATTENTION (GQA) DEEP DIVE
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: GROUPED-QUERY ATTENTION")
print("=" * 70)

print("""
GQA reduces memory by sharing Key and Value projections across head groups.

Standard Multi-Head Attention (MHA):
  - n_heads Query projections
  - n_heads Key projections
  - n_heads Value projections

Grouped-Query Attention (GQA):
  - n_heads Query projections
  - n_kv_heads Key projections  (n_kv_heads < n_heads)
  - n_kv_heads Value projections

Multi-Query Attention (MQA):
  - n_heads Query projections
  - 1 Key projection
  - 1 Value projection
""")

def compare_attention_memory(d_model, n_heads, sequence_length):
    """Compare KV cache memory for different attention types."""
    d_head = d_model // n_heads
    
    # KV cache per token per layer = 2 (K and V) × d_head × n_kv_heads
    # For MHA: n_kv_heads = n_heads
    # For GQA: n_kv_heads < n_heads
    # For MQA: n_kv_heads = 1
    
    results = {}
    for name, n_kv_heads in [("MHA", n_heads), ("GQA (8 groups)", 8), ("MQA", 1)]:
        kv_cache_per_token = 2 * d_head * n_kv_heads * 2  # 2 bytes for FP16
        total_cache = kv_cache_per_token * sequence_length
        results[name] = total_cache
    
    return results

print("\nKV Cache Memory Comparison (Llama 70B-style, per layer):")
print("-" * 60)

kv_comparison = compare_attention_memory(d_model=8192, n_heads=64, sequence_length=8192)
mha_cache = kv_comparison["MHA"]

print(f"Sequence length: 8192 tokens")
print(f"{'Attention Type':<20} {'KV Cache (MB)':>15} {'Savings':>10}")
print("-" * 50)
for name, cache in kv_comparison.items():
    savings = (1 - cache/mha_cache) * 100 if cache < mha_cache else 0
    print(f"{name:<20} {cache/1e6:>14.1f}MB {savings:>9.0f}%")


# =============================================================================
# PART 8: PYTORCH INSPECTION (for when you have it)
# =============================================================================

print("\n" + "=" * 70)
print("PART 8: PYTORCH MODEL INSPECTION")
print("=" * 70)

pytorch_code = '''
"""
When you have PyTorch installed, use this to inspect models:
"""

import torch
import torch.nn as nn

# Create a simple transformer-like model
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, n_layers=4, n_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.lm_head(x)

model = SimpleTransformer()

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Inspect each layer's parameters
print("\\nParameter breakdown:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape} = {param.numel():,}")

# Memory usage
param_memory = total_params * 4 / (1024**2)  # FP32 in MB
print(f"\\nModel size (FP32): {param_memory:.1f} MB")

# Using torchinfo for detailed summary (pip install torchinfo)
# from torchinfo import summary
# summary(model, input_size=(1, 128), dtypes=[torch.long])
'''

print(pytorch_code)


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Key Takeaways:

1. PARAMETERS = WEIGHTS + BIASES
   - Linear layer: (in × out) + out parameters
   - Every learned value is a parameter

2. WHERE PARAMETERS LIVE
   - Embedding: vocab × d_model
   - Attention: ~4 × d_model² per block
   - MLP: ~8 × d_model² per block (with d_ff = 4 × d_model)
   - Layer norms: 2 × d_model per block

3. QUICK ESTIMATE
   - Parameters ≈ 12 × n_layers × d_model²

4. MEMORY = PARAMS × PRECISION
   - FP32: 4 bytes    FP16: 2 bytes
   - INT8: 1 byte     INT4: 0.5 bytes

5. READING MODEL CARDS
   - d_model: richness of representations
   - n_layers: depth of reasoning
   - n_heads vs n_kv_heads: GQA memory savings
   - context length: how much text at once

6. SCALING TRADE-OFFS
   - Bigger = smarter but slower and costlier
   - Quantization enables running large models on smaller hardware
""")
