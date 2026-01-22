"""
RLHF (Reinforcement Learning from Human Feedback) Demo
=======================================================

This demo illustrates the key concepts of RLHF:
1. Verifiable vs Unverifiable tasks
2. Preference data collection
3. Reward model training (conceptual)
4. The RL optimization loop
5. KL penalty and reward hacking
6. DPO as an alternative

No actual training - just showing the concepts clearly.
"""

import random
import math

# =============================================================================
# PART 1: VERIFIABLE VS UNVERIFIABLE TASKS
# =============================================================================

print("=" * 70)
print("PART 1: VERIFIABLE VS UNVERIFIABLE TASKS")
print("=" * 70)

print("""
This distinction is CRUCIAL for understanding when you need RLHF.

┌─────────────────────────────────────────────────────────────────────┐
│  VERIFIABLE TASKS                                                   │
│  Computer can check: RIGHT or WRONG                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Examples:                                                          │
│  • Math: "What is 15 × 23?" → 345 ✓ or 355 ✗                       │
│  • Coding: "Write a function..." → Tests pass ✓ or fail ✗          │
│  • Factual: "Capital of France?" → Paris ✓ or London ✗             │
│                                                                     │
│  → Can use AUTOMATED reward (no humans needed!)                     │
│  → Scales infinitely                                                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  UNVERIFIABLE TASKS                                                 │
│  Humans must judge: Which is BETTER?                                │
├─────────────────────────────────────────────────────────────────────┤
│  Examples:                                                          │
│  • Writing: "Write a poem about spring"                             │
│  • Advice: "How should I handle this situation?"                    │
│  • Explanation: "Explain quantum computing"                         │
│                                                                     │
│  → Need HUMAN preferences (or AI trained on human preferences)      │
│  → Limited by labeling capacity                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# Demonstrate verifiable task
print("\n📊 Verifiable Task Example:")
print("-" * 40)

def check_math(response, correct_answer):
    """Computer can verify math answers automatically."""
    try:
        # Extract number from response
        numbers = [int(s) for s in response.split() if s.isdigit()]
        if numbers and numbers[-1] == correct_answer:
            return "✅ CORRECT", 1.0
        return "❌ WRONG", 0.0
    except:
        return "❌ WRONG", 0.0

prompt = "What is 15 × 23?"
correct = 345

responses = [
    "The answer is 345.",
    "15 times 23 equals 355.",  # Wrong
    "Let me calculate: 345",
]

for resp in responses:
    result, reward = check_math(resp, correct)
    print(f"  Response: '{resp}'")
    print(f"  Result: {result}, Reward: {reward}\n")

# Demonstrate unverifiable task
print("\n📊 Unverifiable Task Example:")
print("-" * 40)

prompt = "Write a haiku about coding"
responses = [
    ("Silent keystrokes fall\nBugs emerge from nested loops\nCoffee grows cold now", "Evocative, follows 5-7-5"),
    ("Code runs on machines\nComputers process the bits\nPrograms execute", "Technically accurate but bland"),
    ("I love to code so much\nIt is really fun\nPython is the best", "Doesn't follow haiku structure"),
]

print(f"  Prompt: '{prompt}'\n")
for i, (resp, note) in enumerate(responses):
    print(f"  Response {chr(65+i)}: \"{resp}\"")
    print(f"  Note: {note}\n")

print("  ❓ Which is best? A computer can't decide - we need human judgment!")


# =============================================================================
# PART 2: PREFERENCE DATA COLLECTION
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 2: PREFERENCE DATA COLLECTION")
print("=" * 70)

print("""
The Wine Tasting Analogy:
─────────────────────────
You don't need tasters to EXPLAIN why they prefer Wine A.
You just need their CHOICE. Patterns emerge from thousands of comparisons.

Process:
1. Give prompt to SFT model
2. Generate two different responses (using temperature/sampling)
3. Human picks which is better
4. Store (prompt, winner, loser) tuple
""")

# Simulated preference data
preference_data = [
    {
        "prompt": "Explain why the sky is blue",
        "response_a": "The sky appears blue due to Rayleigh scattering, where shorter wavelengths of light are scattered more than longer wavelengths by atmospheric molecules.",
        "response_b": "When sunlight hits our atmosphere, blue light bounces around more than other colors because it travels in smaller waves. Think of it like how small balls bounce more in a pinball machine.",
        "winner": "B",
        "reason": "Clearer, good analogy"
    },
    {
        "prompt": "How do I make friends as an adult?",
        "response_a": "Making friends as an adult involves: 1) Join clubs or groups related to your interests. 2) Be consistent - show up regularly. 3) Initiate plans, don't just wait. 4) Be genuinely curious about others.",
        "response_b": "Friendship formation in adulthood is contingent upon repeated exposure and shared activities, as per sociological research on relationship development...",
        "winner": "A",
        "reason": "Practical, actionable, warm"
    },
    {
        "prompt": "Is it OK to lie sometimes?",
        "response_a": "Lying is always wrong. The truth is sacred and must be told at all times regardless of consequences.",
        "response_b": "This is a nuanced question. Most ethicists agree that honesty is generally important, but there are cases (like protecting someone from harm) where the ethics become more complex. What's the context you're thinking about?",
        "winner": "B",
        "reason": "Nuanced, invites dialogue"
    }
]

print("\n📊 Example Preference Comparisons:")
print("-" * 50)

for i, pref in enumerate(preference_data):
    print(f"\n{'─'*60}")
    print(f"Comparison {i+1}")
    print(f"{'─'*60}")
    print(f"Prompt: \"{pref['prompt']}\"")
    print(f"\nResponse A: \"{pref['response_a'][:100]}...\"")
    print(f"\nResponse B: \"{pref['response_b'][:100]}...\"")
    print(f"\n👤 Human chose: {pref['winner']} ({pref['reason']})")

print(f"""
\n📈 Scale of Preference Collection:
┌──────────────────────┬─────────────────┐
│ Company              │ Comparisons     │
├──────────────────────┼─────────────────┤
│ OpenAI (InstructGPT) │ ~50,000         │
│ Anthropic (Claude)   │ ~200,000+       │
│ OpenAssistant        │ ~100,000        │
└──────────────────────┴─────────────────┘
""")


# =============================================================================
# PART 3: REWARD MODEL (CONCEPTUAL)
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: REWARD MODEL - THE AUTOMATIC JUDGE")
print("=" * 70)

print("""
Goal: Build a model that PREDICTS human preferences.

The Analogy:
────────────
After surveying 50,000 customers about coffee preferences, you build
an AI that predicts "customers will prefer drink A over drink B"
without asking them. Now you have UNLIMITED free judging!

Architecture:
┌─────────────────────────────────────────────┐
│  Input: prompt + response                   │
│              ↓                              │
│  [Transformer layers - same as LLM]         │
│              ↓                              │
│  Final hidden state                         │
│              ↓                              │
│  Linear layer → single number (score)       │
└─────────────────────────────────────────────┘
""")

# Simulated reward model
class SimulatedRewardModel:
    """
    A fake reward model that demonstrates the concept.
    Real reward models learn these patterns from data.
    """
    
    def __init__(self):
        # Patterns learned from human preferences (simplified)
        self.positive_signals = [
            "helpful", "clear", "example", "step", "consider",
            "nuanced", "curious", "practical", "let me", "here's"
        ]
        self.negative_signals = [
            "obviously", "simply", "just", "I cannot", "as an AI",
            "I apologize", "unfortunately", "it depends"
        ]
    
    def score(self, prompt, response):
        """Return a score between 0 and 1."""
        response_lower = response.lower()
        
        # Count signals (simplified scoring)
        pos_count = sum(1 for p in self.positive_signals if p in response_lower)
        neg_count = sum(1 for n in self.negative_signals if n in response_lower)
        
        # Base score with some randomness
        base = 0.5 + (pos_count * 0.1) - (neg_count * 0.1)
        noise = random.uniform(-0.05, 0.05)
        
        return max(0.0, min(1.0, base + noise))

reward_model = SimulatedRewardModel()

print("\n📊 Reward Model Scoring Examples:")
print("-" * 50)

test_cases = [
    ("Explain recursion", "Recursion is when a function calls itself. Here's a helpful example: imagine Russian nesting dolls, each containing a smaller version of itself."),
    ("Explain recursion", "Recursion is simply a function calling itself. Obviously, you just need to understand the base case."),
    ("Should I learn Python?", "Python is a great first language! Here's a practical step-by-step approach: 1) Start with basics..."),
    ("Should I learn Python?", "I cannot make that decision for you. It depends on many factors. I apologize but I don't know your situation."),
]

for prompt, response in test_cases:
    score = reward_model.score(prompt, response)
    print(f"\nPrompt: \"{prompt}\"")
    print(f"Response: \"{response[:70]}...\"")
    print(f"Score: {score:.3f} {'⭐' * int(score * 5)}")


# =============================================================================
# PART 4: REWARD MODEL TRAINING
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 4: HOW THE REWARD MODEL LEARNS")
print("=" * 70)

print("""
Training Objective: Given (winner, loser) pair, winner should score HIGHER.

Loss Function (Bradley-Terry model):
    loss = -log(sigmoid(score_winner - score_loser))

Intuition:
• If winner >> loser in score → loss is LOW (good!)
• If winner ≈ loser in score → loss is MEDIUM
• If winner < loser in score → loss is HIGH (bad!)
""")

def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + math.exp(-x))

def reward_model_loss(score_winner, score_loser):
    """Compute the Bradley-Terry loss."""
    diff = score_winner - score_loser
    return -math.log(sigmoid(diff) + 1e-10)  # Add small epsilon for numerical stability

print("\n📊 Loss Function Behavior:")
print("-" * 50)
print(f"{'Winner Score':<14} {'Loser Score':<14} {'Difference':<12} {'Loss':<10}")
print("-" * 50)

test_scores = [
    (0.9, 0.3),  # Clear winner
    (0.7, 0.5),  # Moderate difference
    (0.6, 0.6),  # Tie
    (0.4, 0.7),  # Wrong order
    (0.2, 0.9),  # Very wrong
]

for winner, loser in test_scores:
    diff = winner - loser
    loss = reward_model_loss(winner, loser)
    status = "✓ Good" if diff > 0.1 else "⚠ Close" if diff > -0.1 else "✗ Wrong"
    print(f"{winner:<14.1f} {loser:<14.1f} {diff:<12.1f} {loss:<10.3f} {status}")

print("""
\nWhat the Reward Model Learns (from thousands of comparisons):
┌──────────────────────────────┬──────────────────────────────┐
│ Humans Prefer                │ Over                         │
├──────────────────────────────┼──────────────────────────────┤
│ Direct answers               │ Evasive non-answers          │
│ Accurate information         │ Confident but wrong          │
│ Appropriate length           │ Too short or too verbose     │
│ Clear explanations           │ Jargon-heavy confusion       │
│ Admitting uncertainty        │ Hallucinating confidently    │
│ Safe responses               │ Harmful content              │
└──────────────────────────────┴──────────────────────────────┘
""")


# =============================================================================
# PART 5: RL TRAINING LOOP
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: THE RL TRAINING LOOP")
print("=" * 70)

print("""
Now we use the reward model to improve the language model!

The Dog Training Analogy:
─────────────────────────
• Dog tries an action
• Gets a treat if successful  
• Learns to do more of what earns treats
• Eventually performs tricks reliably

The LLM is the dog. Good responses get treats (high reward).
""")

print("""
The Training Loop:
┌─────────────────────────────────────────────────────────────────────┐
│  for prompt in training_prompts:                                    │
│      response = model.generate(prompt)     # 1. Generate response   │
│      reward = reward_model(prompt, response) # 2. Score it          │
│      rl_update(model, reward)              # 3. Update weights      │
└─────────────────────────────────────────────────────────────────────┘

RL Vocabulary Mapping:
┌────────────────┬───────────────────────────────────────┐
│ RL Term        │ In RLHF                               │
├────────────────┼───────────────────────────────────────┤
│ Agent          │ The language model                    │
│ Environment    │ The conversation                      │
│ State          │ Prompt + tokens generated so far      │
│ Action         │ Choosing the next token               │
│ Reward         │ Reward model score (at the end)       │
│ Policy         │ Model's probability over next tokens  │
└────────────────┴───────────────────────────────────────┘
""")

# Simulate RL training progress
print("\n📊 Simulated RL Training Progress:")
print("-" * 50)

random.seed(42)
for epoch in range(1, 6):
    # Simulate improving average reward over training
    base_reward = 0.4 + (epoch * 0.08)
    rewards = [base_reward + random.uniform(-0.1, 0.1) for _ in range(10)]
    avg_reward = sum(rewards) / len(rewards)
    
    bar = "█" * int(avg_reward * 30)
    print(f"Epoch {epoch}: Avg Reward = {avg_reward:.3f} |{bar}")

print("""
\n💡 The Credit Assignment Problem:
   Reward comes at the END, but decisions happen at EVERY token.
   
   "The | capital | of | France | is | Paris | ."
                                              ↑
                                      Reward given here
   
   Which tokens contributed to the good score?
   PPO handles this through "advantage estimation".
""")


# =============================================================================
# PART 6: KL PENALTY - PREVENTING REWARD HACKING
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: KL PENALTY - DON'T DRIFT TOO FAR")
print("=" * 70)

print("""
The Problem: If you ONLY optimize for reward, the model finds exploits!

The Barista Analogy:
────────────────────
The taste predictor really likes foam.
Solution? ALL FOAM, NO COFFEE! 
Scores high, but not what we wanted!

Real Examples of Reward Hacking:
• Reward model likes confident responses → Model becomes overconfident
• Reward model likes detailed responses → Model adds meaningless padding
• Reward model likes safe responses → Model refuses EVERYTHING
""")

print("""
The Solution: KL Penalty
────────────────────────
Keep the model CLOSE to the SFT version.

    total_reward = reward_score - β × KL_penalty
                        ↑                ↑
                   what we want    penalty for drifting

KL divergence measures how DIFFERENT two distributions are.
High KL = model has drifted far from the SFT version = BAD
""")

# Simulate KL penalty effect
print("\n📊 Effect of KL Penalty (β):")
print("-" * 50)

def simulate_training_outcome(beta):
    """Simulate how different β values affect training."""
    if beta < 0.01:
        return "Reward hacking! All foam, no coffee", 0.95, 0.3
    elif beta < 0.05:
        return "Good balance, model improves", 0.75, 0.7
    elif beta < 0.2:
        return "Conservative, steady improvement", 0.65, 0.8
    else:
        return "Too conservative, barely changes", 0.52, 0.95

print(f"{'β Value':<10} {'Outcome':<45} {'Reward':<10} {'Quality'}")
print("-" * 75)

for beta in [0.001, 0.01, 0.05, 0.1, 0.3]:
    outcome, reward, quality = simulate_training_outcome(beta)
    print(f"{beta:<10.3f} {outcome:<45} {reward:<10.2f} {'⭐' * int(quality * 5)}")

print("""
\n🎯 Goodhart's Law:
   "When a measure becomes a target, it ceases to be a good measure."
   
   Students optimize for TEST SCORES, not actual LEARNING.
   Models optimize for REWARD SCORES, not actual HELPFULNESS.
   
   The KL penalty helps prevent this by keeping the model grounded.
""")


# =============================================================================
# PART 7: DPO - THE SIMPLER ALTERNATIVE
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: DPO - DIRECT PREFERENCE OPTIMIZATION")
print("=" * 70)

print("""
The Insight: What if we skip the reward model entirely?

RLHF Pipeline:
    preferences → reward model → RL training → better model
    (complex!)

DPO Pipeline:  
    preferences ────────────────────────────→ better model
    (simple!)

The Analogy:
────────────
• RLHF: Build an AI taste predictor, then use it to train the barista
• DPO: Just show the barista pairs of drinks and say which customers preferred

DPO directly trains the model so that:
• Winners become MORE likely
• Losers become LESS likely
• But not too far from the original (implicit KL penalty!)
""")

print("""
Comparison:
┌─────────────────┬─────────────────────┬─────────────────────┐
│ Aspect          │ RLHF (PPO)          │ DPO                 │
├─────────────────┼─────────────────────┼─────────────────────┤
│ Complexity      │ High (3 stages)     │ Low (fine-tuning)   │
│ Stability       │ Can be unstable     │ Standard loss       │
│ Compute         │ RL + reward model   │ Just fine-tuning    │
│ Performance     │ Strong              │ Comparable          │
│ Flexibility     │ Reusable RM         │ Direct optimization │
└─────────────────┴─────────────────────┴─────────────────────┘

Current Trend: DPO is increasingly popular due to simplicity!
""")


# =============================================================================
# PART 8: CONSTITUTIONAL AI / RLAIF
# =============================================================================

print("\n" + "=" * 70)
print("PART 8: CONSTITUTIONAL AI - AI AS JUDGE")
print("=" * 70)

print("""
The Human Bottleneck:
─────────────────────
RLHF needs human labels. Humans are:
• Expensive
• Slow  
• Limited in capacity

The Sommelier Analogy:
Having professional sommeliers taste every wine pair.
Accurate, but you can only do so many per day.

The Solution: RLAIF (RL from AI Feedback)
─────────────────────────────────────────
Use a capable AI model to generate preferences!

Constitutional AI (Anthropic's approach):
1. Write a "constitution" (principles)
2. AI critiques and improves responses based on principles
3. Use (original, improved) as preference pairs
4. Train with DPO

Example Constitution Principles:
┌─────────────────────────────────────────────────────────────────────┐
│ • Be helpful and provide accurate information                       │
│ • Don't assist with harmful or illegal activities                   │
│ • Acknowledge uncertainty when appropriate                          │
│ • Be respectful and avoid offensive content                         │
│ • Consider multiple perspectives on controversial topics            │
└─────────────────────────────────────────────────────────────────────┘

The Analogy:
────────────
• Humans write the LAW (constitution)
• AI is the army of LAWYERS applying it consistently to millions of cases
• Humans define values, AI scales the application
""")

print("""
Benefits of Constitutional AI:
┌─────────────────┬───────────────────────────────────────────────┐
│ Scalable        │ AI generates millions of comparisons          │
│ Consistent      │ Same principles applied uniformly             │
│ Transparent     │ Principles are explicit and auditable         │
│ Iterable        │ Update principles, regenerate preferences     │
│ Cheap           │ No human labelers needed at scale             │
└─────────────────┴───────────────────────────────────────────────┘
""")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: RLHF IN A NUTSHELL")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│  RLHF (Reinforcement Learning from Human Feedback)                  │
├─────────────────────────────────────────────────────────────────────┤
│  Goal:      Generate responses humans PREFER                        │
│             (correct, accurate, safe, helpful, polite)              │
│                                                                     │
│  Key Distinction:                                                   │
│  • Verifiable tasks (math, code) → Computer can reward              │
│  • Unverifiable tasks (writing) → Need human preferences            │
│                                                                     │
│  Three Steps:                                                       │
│  1. Collect preferences (100K-1M comparisons)                       │
│  2. Train reward model (automatic judge)                            │
│  3. RL fine-tuning (maximize reward with KL penalty)                │
│                                                                     │
│  Algorithms: PPO (original), DPO (simpler), GRPO (group-based)      │
│                                                                     │
│  Key Insight: Reward model scales human preferences INFINITELY      │
│                                                                     │
│  Danger:     Reward hacking (Goodhart's Law)                        │
│  Solution:   KL penalty keeps model grounded                        │
│                                                                     │
│  Scaling:    Constitutional AI / RLAIF uses AI to judge AI          │
└─────────────────────────────────────────────────────────────────────┘

The Complete Post-Training Pipeline:
────────────────────────────────────
Base Model (completion)
     ↓
┌─────────────────┐
│  SFT            │  "Here are example responses"
│                 │  Analogy: Showing example emails
└─────────────────┘
     ↓
┌─────────────────┐
│  RLHF/DPO      │  "This response is better"
│                 │  Analogy: Manager gives feedback
└─────────────────┘
     ↓
Aligned Model (helpful assistant)
""")

print("\n✅ Demo complete! Ready for Evaluation and System Design topics.")
