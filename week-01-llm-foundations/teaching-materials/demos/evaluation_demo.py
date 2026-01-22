"""
LLM Evaluation Demo
==================

This demo shows how different evaluation methods work:
1. Perplexity calculation (traditional evaluation)
2. Benchmark evaluation (task-specific)
3. ELO rating system (crowdsourced comparison)

Run: python evaluation_demo.py
"""

import math
import random
from typing import List, Dict, Tuple

print("=" * 60)
print("LLM EVALUATION DEMO")
print("=" * 60)


# =============================================================================
# PART 1: PERPLEXITY
# =============================================================================

print("\n" + "=" * 60)
print("PART 1: PERPLEXITY - How surprised is the model?")
print("=" * 60)

def calculate_perplexity(token_probabilities: List[float]) -> float:
    """
    Calculate perplexity from a list of token probabilities.
    
    Perplexity = exp(-1/N × Σ log P(token_i))
    
    Lower perplexity = model was more confident = better predictions
    """
    n = len(token_probabilities)
    
    # Sum of log probabilities
    log_prob_sum = sum(math.log(p) for p in token_probabilities)
    
    # Average negative log probability
    avg_neg_log_prob = -log_prob_sum / n
    
    # Perplexity is exp of that
    perplexity = math.exp(avg_neg_log_prob)
    
    return perplexity


print("\n--- Example: 'How are you doing?' ---")
print()

# Simulating what a model might predict for each token
# Format: (context, actual_next_token, probability_model_gave_to_correct_token)
predictions = [
    ("How", "are", 0.28),
    ("How are", "you", 0.56),
    ("How are you", "doing", 0.93),
    ("How are you doing", "?", 0.58),
]

print("Token-by-token predictions:")
print("-" * 50)
for context, token, prob in predictions:
    print(f"  Context: '{context}'")
    print(f"  Actual next token: '{token}'")
    print(f"  Model's probability for '{token}': {prob}")
    print()

# Extract just the probabilities
probabilities = [p[2] for p in predictions]

# Calculate perplexity step by step
print("Calculation:")
print("-" * 50)
log_probs = [math.log(p) for p in probabilities]
print(f"  log(0.28) = {log_probs[0]:.3f}")
print(f"  log(0.56) = {log_probs[1]:.3f}")
print(f"  log(0.93) = {log_probs[2]:.3f}")
print(f"  log(0.58) = {log_probs[3]:.3f}")
print()

log_sum = sum(log_probs)
print(f"  Sum of log probs: {log_sum:.3f}")

avg_neg = -log_sum / len(log_probs)
print(f"  Average negative log prob: {avg_neg:.3f}")

ppl = math.exp(avg_neg)
print(f"  Perplexity = exp({avg_neg:.3f}) = {ppl:.2f}")

print()
print(f"✓ Perplexity: {ppl:.2f}")
print()
print("Interpretation: The model was choosing between ~2 equally")
print("likely options on average. Pretty good!")


# Compare different scenarios
print("\n--- Comparing Different Scenarios ---")
print()

scenarios = [
    ("Perfect prediction", [0.99, 0.99, 0.99, 0.99]),
    ("Very confident", [0.8, 0.9, 0.85, 0.75]),
    ("Somewhat confident", [0.5, 0.6, 0.4, 0.55]),
    ("Confused", [0.1, 0.15, 0.2, 0.1]),
    ("Random guessing (vocab=50000)", [1/50000] * 4),
]

print(f"{'Scenario':<35} {'Perplexity':>12}")
print("-" * 50)
for name, probs in scenarios:
    ppl = calculate_perplexity(probs)
    print(f"{name:<35} {ppl:>12.2f}")

print()
print("Key insight: Lower perplexity = better predictions")
print("But perplexity doesn't tell us if the content is USEFUL!")


# =============================================================================
# PART 2: BENCHMARK EVALUATION
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: BENCHMARKS - Task-specific evaluation")
print("=" * 60)


def create_benchmark(name: str, questions: List[Dict]) -> Dict:
    """Create a simple benchmark dataset."""
    return {"name": name, "questions": questions}


def evaluate_on_benchmark(model_answers: List[str], benchmark: Dict) -> float:
    """
    Evaluate model answers against benchmark.
    Returns accuracy (0-100%).
    """
    correct = 0
    total = len(benchmark["questions"])
    
    for i, q in enumerate(benchmark["questions"]):
        expected = q["answer"].lower().strip()
        actual = model_answers[i].lower().strip()
        if expected == actual:
            correct += 1
    
    return (correct / total) * 100


# Create mini benchmarks
print("\n--- Mini Benchmark: Common-sense Reasoning ---")
print()

commonsense_benchmark = create_benchmark("MiniHellaSwag", [
    {
        "question": "The trophy doesn't fit in the suitcase because it is too large. What is too large?",
        "choices": ["the trophy", "the suitcase"],
        "answer": "the trophy"
    },
    {
        "question": "The ball broke the window because it was fragile. What was fragile?",
        "choices": ["the ball", "the window"],
        "answer": "the window"
    },
    {
        "question": "The dog chased the cat because it was scared. What was scared?",
        "choices": ["the dog", "the cat"],
        "answer": "the cat"
    },
])

print("Questions:")
for i, q in enumerate(commonsense_benchmark["questions"]):
    print(f"  {i+1}. {q['question']}")
    print(f"     Choices: {q['choices']}")
    print(f"     Correct: {q['answer']}")
    print()

# Simulate different model performances
print("Simulated Model Answers:")
print("-" * 50)

model_a_answers = ["the trophy", "the window", "the cat"]  # All correct
model_b_answers = ["the suitcase", "the window", "the dog"]  # 1 correct
model_c_answers = ["the trophy", "the ball", "the cat"]  # 2 correct

print("  Model A: ['the trophy', 'the window', 'the cat']")
print("  Model B: ['the suitcase', 'the window', 'the dog']")
print("  Model C: ['the trophy', 'the ball', 'the cat']")
print()

score_a = evaluate_on_benchmark(model_a_answers, commonsense_benchmark)
score_b = evaluate_on_benchmark(model_b_answers, commonsense_benchmark)
score_c = evaluate_on_benchmark(model_c_answers, commonsense_benchmark)

print("Results:")
print(f"  Model A: {score_a:.1f}%")
print(f"  Model B: {score_b:.1f}%")
print(f"  Model C: {score_c:.1f}%")


print("\n--- Mini Benchmark: Math (GSM8K-style) ---")
print()

math_benchmark = create_benchmark("MiniGSM8K", [
    {
        "question": "A train travels 60 mph for 3 hours. How far does it travel?",
        "answer": "180"
    },
    {
        "question": "If you have 24 apples and give away 1/3, how many do you have left?",
        "answer": "16"
    },
    {
        "question": "A rectangle has width 5 and length 8. What is its area?",
        "answer": "40"
    },
])

print("Questions:")
for i, q in enumerate(math_benchmark["questions"]):
    print(f"  {i+1}. {q['question']}")
    print(f"     Answer: {q['answer']}")
    print()

# Simulate answers
math_answers_good = ["180", "16", "40"]
math_answers_bad = ["180", "8", "13"]  # Common mistakes

score_good = evaluate_on_benchmark(math_answers_good, math_benchmark)
score_bad = evaluate_on_benchmark(math_answers_bad, math_benchmark)

print("Results:")
print(f"  Good model: {score_good:.1f}%")
print(f"  Bad model:  {score_bad:.1f}%")

print()
print("Key insight: Benchmarks test SPECIFIC capabilities")
print("High benchmark score ≠ generally intelligent (Goodhart's Law!)")


# =============================================================================
# PART 3: ELO RATING SYSTEM
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: ELO RATING - Crowdsourced comparison (LMArena)")
print("=" * 60)


class ELOSystem:
    """
    Simple ELO rating system for comparing models.
    Same system used by chess and LMArena.
    """
    
    def __init__(self, k_factor: int = 32):
        self.ratings: Dict[str, float] = {}
        self.k_factor = k_factor
        self.history: List[str] = []
    
    def add_model(self, name: str, initial_rating: float = 1500):
        """Add a new model with initial rating."""
        self.ratings[name] = initial_rating
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.
        Returns probability of A winning (0 to 1).
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, winner: str, loser: str, tie: bool = False):
        """
        Update ratings after a match.
        winner/loser can be same if tie=True
        """
        rating_a = self.ratings[winner]
        rating_b = self.ratings[loser]
        
        # Expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a
        
        # Actual scores
        if tie:
            actual_a = 0.5
            actual_b = 0.5
        else:
            actual_a = 1
            actual_b = 0
        
        # Update ratings
        new_rating_a = rating_a + self.k_factor * (actual_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (actual_b - expected_b)
        
        self.ratings[winner] = new_rating_a
        self.ratings[loser] = new_rating_b
        
        # Record history
        result = "tie" if tie else f"{winner} beat {loser}"
        self.history.append(
            f"{result}: {winner} {rating_a:.0f}→{new_rating_a:.0f}, "
            f"{loser} {rating_b:.0f}→{new_rating_b:.0f}"
        )
        
        return new_rating_a, new_rating_b
    
    def get_leaderboard(self) -> List[Tuple[str, float]]:
        """Get models sorted by rating."""
        return sorted(self.ratings.items(), key=lambda x: -x[1])


print("\n--- Setting Up the Tournament ---")
print()

# Create ELO system with some models
elo = ELOSystem(k_factor=32)
elo.add_model("GPT-4", 1500)
elo.add_model("Claude-3", 1500)
elo.add_model("Llama-3", 1500)
elo.add_model("Gemini", 1500)

print("Initial ratings (all start at 1500):")
for model, rating in elo.get_leaderboard():
    print(f"  {model}: {rating:.0f}")


print("\n--- Simulating User Comparisons ---")
print()

# Simulate some comparisons (like what happens on LMArena)
comparisons = [
    ("GPT-4", "Llama-3", False),      # GPT-4 wins
    ("Claude-3", "Gemini", False),     # Claude wins
    ("GPT-4", "Claude-3", False),      # GPT-4 wins
    ("Claude-3", "Llama-3", False),    # Claude wins
    ("Gemini", "Llama-3", False),      # Gemini wins
    ("GPT-4", "Gemini", False),        # GPT-4 wins
    ("Claude-3", "GPT-4", False),      # Claude wins (upset!)
    ("Llama-3", "Gemini", True),       # Tie
]

print("Matches (simulated user preferences):")
print("-" * 50)
for winner, loser, is_tie in comparisons:
    elo.update_ratings(winner, loser, tie=is_tie)
    print(f"  {elo.history[-1]}")

print("\n--- Final Leaderboard ---")
print()
print(f"{'Rank':<6} {'Model':<15} {'Rating':>8}")
print("-" * 32)
for i, (model, rating) in enumerate(elo.get_leaderboard(), 1):
    print(f"{i:<6} {model:<15} {rating:>8.1f}")


print("\n--- How ELO Handles Upsets ---")
print()

# Demonstrate upset mechanics
demo_elo = ELOSystem(k_factor=32)
demo_elo.add_model("Favorite", 1600)
demo_elo.add_model("Underdog", 1400)

print("Setup: Favorite (1600) vs Underdog (1400)")
print()

expected = demo_elo.expected_score(1600, 1400)
print(f"Expected win probability for Favorite: {expected:.1%}")
print()

# Scenario 1: Favorite wins (expected)
print("Scenario 1: Favorite wins (expected outcome)")
demo_elo.ratings = {"Favorite": 1600, "Underdog": 1400}  # Reset
demo_elo.update_ratings("Favorite", "Underdog")
print(f"  Favorite: 1600 → {demo_elo.ratings['Favorite']:.1f} (+{demo_elo.ratings['Favorite']-1600:.1f})")
print(f"  Underdog: 1400 → {demo_elo.ratings['Underdog']:.1f} ({demo_elo.ratings['Underdog']-1400:.1f})")
print()

# Scenario 2: Underdog wins (upset!)
print("Scenario 2: Underdog wins (UPSET!)")
demo_elo.ratings = {"Favorite": 1600, "Underdog": 1400}  # Reset
demo_elo.update_ratings("Underdog", "Favorite")
print(f"  Underdog: 1400 → {demo_elo.ratings['Underdog']:.1f} (+{demo_elo.ratings['Underdog']-1400:.1f})")
print(f"  Favorite: 1600 → {demo_elo.ratings['Favorite']:.1f} ({demo_elo.ratings['Favorite']-1600:.1f})")
print()

print("Key insight: Upsets cause BIGGER rating changes!")
print("This is why ELO self-corrects over time.")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: EVALUATION METHODS COMPARED")
print("=" * 60)

print("""
┌─────────────────┬────────────────────┬─────────────────────┐
│ Method          │ What It Measures   │ Limitation          │
├─────────────────┼────────────────────┼─────────────────────┤
│ Perplexity      │ Prediction quality │ Doesn't measure     │
│                 │ (fluency)          │ usefulness          │
├─────────────────┼────────────────────┼─────────────────────┤
│ Benchmarks      │ Specific tasks     │ Can be gamed        │
│                 │ (math, code, etc.) │ (Goodhart's Law)    │
├─────────────────┼────────────────────┼─────────────────────┤
│ ELO / LMArena   │ Human preference   │ Limited to site     │
│                 │ (blind comparison) │ visitors            │
└─────────────────┴────────────────────┴─────────────────────┘

Key takeaway: No single metric is enough!
Real evaluation uses ALL methods together.
""")

print("=" * 60)
print("END OF DEMO")
print("=" * 60)
