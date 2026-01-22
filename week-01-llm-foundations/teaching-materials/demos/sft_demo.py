"""
SFT (Supervised Fine-Tuning) Demo
=================================

This demo illustrates the key concepts of SFT:
1. Demonstration data formats
2. Loss masking (only learn from responses)
3. The transformation from completion to instruction-following

No actual training - just showing the concepts clearly.
"""

import json

# =============================================================================
# PART 1: DEMONSTRATION DATA FORMATS
# =============================================================================

print("=" * 70)
print("PART 1: DEMONSTRATION DATA FORMATS")
print("=" * 70)

# Format 1: Simple instruction/output pairs
simple_format = [
    {
        "instruction": "What is the capital of France?",
        "output": "The capital of France is Paris."
    },
    {
        "instruction": "Explain photosynthesis to a 5-year-old",
        "output": "Plants are like little chefs that make their own food! They use sunlight as their oven, water as one ingredient, and air as another. When they mix these together with the green stuff in their leaves, they make sugar to eat!"
    },
    {
        "instruction": "Write a haiku about programming",
        "output": "Silent keystrokes fall\nBugs emerge from nested loops\nCoffee grows cold now"
    }
]

print("\n📝 Format 1: Simple Instruction/Output Pairs")
print("-" * 50)
for i, example in enumerate(simple_format):
    print(f"\nExample {i+1}:")
    print(f"  Instruction: {example['instruction'][:50]}...")
    print(f"  Output: {example['output'][:50]}...")

# Format 2: Conversation format (multi-turn)
conversation_format = [
    {
        "messages": [
            {"role": "user", "content": "What's the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What's it famous for?"},
            {"role": "assistant", "content": "Paris is famous for the Eiffel Tower, the Louvre museum, its café culture, and being a global center for fashion and cuisine."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Help me write a professional email declining a meeting."},
            {"role": "assistant", "content": "I'd be happy to help! Could you tell me the context - who is the meeting with and why do you need to decline?"},
            {"role": "user", "content": "It's with a vendor, and I have a conflicting priority project deadline."},
            {"role": "assistant", "content": "Here's a professional email:\n\nSubject: Request to Reschedule Our Meeting\n\nDear [Vendor Name],\n\nThank you for scheduling time to meet. Unfortunately, I have a critical project deadline that conflicts with our planned meeting time.\n\nCould we reschedule to next week? I'm available [suggest times].\n\nI apologize for any inconvenience and look forward to connecting soon.\n\nBest regards,\n[Your name]"}
        ]
    }
]

print("\n\n📝 Format 2: Conversation Format (Multi-turn)")
print("-" * 50)
for i, conv in enumerate(conversation_format):
    print(f"\nConversation {i+1}: ({len(conv['messages'])} turns)")
    for msg in conv['messages']:
        role = "👤 User" if msg['role'] == 'user' else "🤖 Assistant"
        content = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']
        print(f"  {role}: {content}")


# =============================================================================
# PART 2: CHAT TEMPLATES
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 2: CHAT TEMPLATES")
print("=" * 70)
print("\nDifferent models use different formats to mark conversation turns.")
print("Using the wrong template = confused model outputs!")

conversation = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
]

# ChatML format (OpenAI style)
def to_chatml(messages):
    result = ""
    for msg in messages:
        result += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return result

# Llama/Alpaca format
def to_llama(messages):
    result = ""
    for i, msg in enumerate(messages):
        if msg['role'] == 'user':
            result += f"[INST] {msg['content']} [/INST] "
        else:
            result += f"{msg['content']}\n"
    return result.strip()

# Simple format
def to_simple(messages):
    result = ""
    for msg in messages:
        role = "Human" if msg['role'] == 'user' else "Assistant"
        result += f"{role}: {msg['content']}\n\n"
    return result.strip()

print("\n🏷️ ChatML Format (OpenAI style):")
print("-" * 40)
print(to_chatml(conversation))

print("\n🦙 Llama/Alpaca Format:")
print("-" * 40)
print(to_llama(conversation))

print("\n📋 Simple Format:")
print("-" * 40)
print(to_simple(conversation))


# =============================================================================
# PART 3: LOSS MASKING - THE KEY DIFFERENCE
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 3: LOSS MASKING - THE KEY DIFFERENCE")
print("=" * 70)

print("""
In pre-training: We compute loss on ALL tokens
In SFT:          We compute loss ONLY on response tokens

Why? We want the model to learn to GENERATE good responses,
     not to GENERATE user questions.
""")

# Simulate tokenization and loss masking
def simulate_loss_masking(prompt, response):
    """Show how loss masking works in SFT."""
    
    # Simple word-level "tokenization" for demonstration
    prompt_tokens = prompt.split()
    response_tokens = response.split()
    all_tokens = prompt_tokens + response_tokens
    
    # Create mask: 0 for prompt, 1 for response
    mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)
    
    return all_tokens, mask

prompt = "User: What is 2+2? Assistant:"
response = "The answer is 4."

tokens, mask = simulate_loss_masking(prompt, response)

print("\n📊 Loss Masking Visualization:")
print("-" * 50)
print("\nTokens and their loss mask:")
print()

# Print in a nice table format
print(f"{'Token':<15} {'Mask':<6} {'Meaning'}")
print("-" * 40)
for token, m in zip(tokens, mask):
    meaning = "← LEARN from this" if m == 1 else "← ignore"
    print(f"{token:<15} {m:<6} {meaning}")

print(f"""
\n💡 Key Insight:
   - Prompt tokens (mask=0): Model sees these but doesn't learn to generate them
   - Response tokens (mask=1): Model learns to generate these given the prompt
""")


# =============================================================================
# PART 4: THE TRANSFORMATION (BEFORE vs AFTER SFT)
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 4: THE TRANSFORMATION (Before vs After SFT)")
print("=" * 70)

examples = [
    {
        "prompt": "I want to learn machine learning. What should I do?",
        "before_sft": "I want to learn machine learning. What should I do if I don't have a math background? This is a question many beginners ask when starting their journey into artificial intelligence and data science...",
        "after_sft": "Great question! Here's a recommended path:\n\n1. Start with Andrew Ng's Machine Learning course on Coursera\n2. Learn Python basics if you haven't already\n3. Practice with scikit-learn on small datasets\n4. Move to deep learning with PyTorch or TensorFlow\n\nWould you like specific resource recommendations for any of these steps?"
    },
    {
        "prompt": "Write a poem about the ocean",
        "before_sft": "Write a poem about the ocean\n\nThe ocean has inspired countless poets throughout history. Some famous ocean poems include 'The Rime of the Ancient Mariner' by Samuel Taylor Coleridge and 'Dover Beach' by Matthew Arnold...",
        "after_sft": "Beneath the sky of endless blue,\nWhere waves compose their rolling tune,\nThe ocean breathes with ancient sighs,\nAnd mirrors back the wandering skies.\n\nSalt and foam and mystery deep,\nWhere secret creatures drift and sleep,\nA world apart, yet part of ours—\nThe sea, keeper of infinite hours."
    },
    {
        "prompt": "Is it safe to eat raw cookie dough?",
        "before_sft": "Is it safe to eat raw cookie dough? Many people enjoy eating raw cookie dough, and this practice has been common for decades. However, health experts have expressed concerns about...",
        "after_sft": "Raw cookie dough is generally not recommended because it contains:\n\n1. **Raw eggs** - Risk of Salmonella\n2. **Raw flour** - Can contain E. coli (flour isn't heat-treated)\n\nIf you love the taste, try 'edible cookie dough' recipes that use heat-treated flour and no eggs. Many stores also sell safe-to-eat cookie dough now!\n\nWant a recipe for safe homemade edible cookie dough?"
    }
]

for i, ex in enumerate(examples):
    print(f"\n{'='*60}")
    print(f"Example {i+1}")
    print(f"{'='*60}")
    print(f"\n📝 Prompt: \"{ex['prompt']}\"")
    print(f"\n❌ BEFORE SFT (base model completes text):")
    print(f"   {ex['before_sft'][:200]}...")
    print(f"\n✅ AFTER SFT (model answers the question):")
    print(f"   {ex['after_sft'][:200]}...")


# =============================================================================
# PART 5: QUALITY VS QUANTITY
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 5: QUALITY VS QUANTITY")
print("=" * 70)

print("""
A key insight from SFT research: QUALITY beats QUANTITY.

Dataset Comparison:
┌─────────────────┬─────────┬───────────┬─────────────────────┐
│ Dataset         │ Size    │ Source    │ Quality             │
├─────────────────┼─────────┼───────────┼─────────────────────┤
│ InstructGPT     │ 13K     │ Experts   │ ⭐⭐⭐⭐⭐ (highest)    │
│ Dolly           │ 15K     │ Employees │ ⭐⭐⭐⭐ (high)        │
│ Alpaca          │ 52K     │ GPT-3.5   │ ⭐⭐⭐ (medium)       │
│ OpenAssistant   │ 161K    │ Crowd     │ ⭐⭐⭐ (variable)     │
│ FLAN            │ 1M+     │ Converted │ ⭐⭐ (mixed)         │
└─────────────────┴─────────┴───────────┴─────────────────────┘

Result: 13K expert examples often outperform 1M+ mixed examples!

Why? The model learns the pattern quickly. What matters is the
pattern being CORRECT and CONSISTENT.

The Analogy:
- Learning to write professional emails from 13 perfect examples
  written by a senior executive
- vs. learning from 1000 emails of varying quality from random
  employees

The 13 perfect examples teach the right pattern. The 1000 mixed
examples confuse with inconsistencies.
""")


# =============================================================================
# PART 6: SFT LIMITATIONS (WHY WE NEED RLHF)
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: WHY SFT ISN'T ENOUGH")
print("=" * 70)

print("""
After SFT, the model follows instructions. But it's NOT ready to deploy.

❌ Problem 1: Can't cover everything
   You can't write demonstrations for every possible question.

❌ Problem 2: Averages over annotator styles
   Different experts write differently → model learns a blend

❌ Problem 3: No preference signal
   SFT can't express "Response A is BETTER than Response B"
""")

print("\n📊 The Preference Problem:")
print("-" * 50)

prompt = "Explain quantum computing"
responses = [
    ("A", "Quantum computing uses qubits that can exist in superposition, allowing parallel computation of multiple states simultaneously through quantum mechanical phenomena like entanglement.", "Technical, accurate, dense"),
    ("B", "Imagine a coin spinning in the air - it's both heads AND tails until it lands. Quantum computers use 'qubits' that work similarly, letting them explore many solutions at once.", "Clear analogy, accessible"),
    ("C", "Quantum computing is a revolutionary paradigm shift in computational methodology that leverages the principles of quantum mechanics, specifically superposition and entanglement, to perform calculations that would be intractable for classical computational systems, enabling exponential speedups for certain problem classes including cryptographic analysis, molecular simulation, and optimization problems.", "Verbose, overwhelming")
]

for letter, response, style in responses:
    print(f"\nResponse {letter} ({style}):")
    print(f"  \"{response[:80]}...\"")

print("""
\n❓ Which is "best"?
   - All are accurate
   - All could be in SFT training data
   - SFT can't express: "B > A > C" (preference ranking)

👉 This is why we need RLHF: to learn PREFERENCES, not just examples.
""")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: SFT IN A NUTSHELL")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│  SFT (Supervised Fine-Tuning)                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Goal:      Completion model → Instruction-following model          │
│                                                                     │
│  Data:      Demonstration data (prompt/response pairs)              │
│             ~10K-100K high-quality examples                         │
│                                                                     │
│  Training:  Same as pre-training + loss masking on responses        │
│             Lower learning rate to preserve pre-trained knowledge   │
│                                                                     │
│  Key:       Quality > Quantity (13K expert > 1M mixed)              │
│                                                                     │
│  Analogy:   Showing a new employee example emails                   │
│                                                                     │
│  Limitation: Can show what's good, can't compare what's BETTER      │
│              → This is why we need RLHF next!                       │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\n✅ Demo complete! Next: rl_demo.py for RLHF concepts.")
