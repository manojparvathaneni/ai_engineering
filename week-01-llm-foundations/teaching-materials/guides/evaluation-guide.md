# LLM Evaluation Guide

## Overview

After building an LLM through pre-training, SFT, and RLHF, we need to answer: **"Is it any good?"**

This turns out to be surprisingly difficult because:
- LLMs are general-purpose (can't test every possible task)
- "Good" is subjective (helpful to whom? for what?)
- Models can game metrics (Goodhart's Law)
- Lab performance ≠ real-world performance

## The Evaluation Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                         EVALUATION                               │
├────────────────────────────────┬────────────────────────────────┤
│      OFFLINE                   │         ONLINE                  │
│   (before deployment)          │     (after deployment)          │
│   "Lab testing"                │     "Real-world testing"        │
├────────────────────────────────┼────────────────────────────────┤
│ 1. Traditional (Perplexity)    │ 1. User feedback (thumbs)       │
│ 2. Task-specific (Benchmarks)  │ 2. Crowdsourcing (LMArena)      │
│ 3. Human evaluation            │                                 │
└────────────────────────────────┴────────────────────────────────┘
```

**Analogy:**
- Offline = Testing a car on a track before selling it
- Online = Seeing how customers actually drive it

---

## 1. Traditional Evaluation: Perplexity

### What Is Perplexity?

LLMs are trained to predict the next token. Perplexity measures: **How surprised is the model by the actual text?**

```
Text: "How are you doing?"

Step by step:
┌─────────────────┬──────────────────┬─────────────────┐
│ Context         │ Actual next word │ Model's P(word) │
├─────────────────┼──────────────────┼─────────────────┤
│ "How"           │ "are"            │ 0.28            │
│ "How are"       │ "you"            │ 0.56            │
│ "How are you"   │ "doing"          │ 0.93            │
│ "How are you d" │ "?"              │ 0.58            │
└─────────────────┴──────────────────┴─────────────────┘
```

If the model predicted high probability for each actual token → low surprise → low perplexity → good!

### The Formula

```
Perplexity(text) = exp( -1/N × Σ log P(token_i | context) )
```

**Intuitive interpretation:** Perplexity ≈ "How many choices was the model confused between?"

| Perplexity | Interpretation |
|------------|----------------|
| 1 | Perfect! Model was 100% sure of every token |
| 10 | Model was choosing between ~10 equally likely options |
| 100 | Model was very confused |
| 1000+ | Model has no idea what's going on |

**Lower perplexity = better predictions**

### Worked Example

```
Perplexity("How are you doing?") 
= exp(-1/4 × (log(0.28) + log(0.56) + log(0.93) + log(0.58)))
= exp(-1/4 × (-1.27 + -0.58 + -0.07 + -0.54))
= exp(0.615)
= 1.85
```

A perplexity of 1.85 means the model was choosing between about 2 equally likely options on average.

### Why Perplexity Is LIMITED

**Critical insight:** Perplexity measures fluency, not usefulness!

```
Prompt: "What's 2+2?"

Response A: "2+2 equals 4."           ← Correct!
Response B: "2+2 equals 5."           ← Wrong!
Response C: "The answer is unclear."  ← Unhelpful!
```

All three responses could have **identical perplexity** if they're fluent English!

Perplexity does NOT measure:
- ❌ Correctness
- ❌ Helpfulness
- ❌ Safety
- ❌ Following instructions

**Analogy:** Grading an essay only on grammar, ignoring whether the content makes sense.

---

## 2. Task-Specific Evaluation: Benchmarks

Since perplexity doesn't tell us if the model is useful, we test specific capabilities.

### Benchmark Categories

| Category | What It Tests | Example Benchmarks |
|----------|---------------|-------------------|
| **Common-sense reasoning** | Understanding everyday situations | PIQA, SIQA, HellaSwag |
| **World knowledge** | Factual recall | TriviaQA, Natural Questions, SQuAD |
| **Mathematical reasoning** | Solving math problems | MATH, GSM8K |
| **Code generation** | Writing working code | HumanEval, MBPP |
| **Composite** | Multiple abilities | MMLU (57 subjects) |

### Example Questions

**Common-sense (HellaSwag):**
```
"The trophy doesn't fit in the suitcase because it is too large. 
What is too large?"
(a) the trophy  ← Correct (requires understanding "it")
(b) the suitcase
```

**World knowledge (TriviaQA):**
```
"Who wrote the play Hamlet?"
Answer: William Shakespeare
```

**Mathematical reasoning (GSM8K):**
```
"If a train travels 60 miles per hour for 3 hours, 
how far does it travel?"
Answer: 180 miles
```

**Code generation (HumanEval):**
```
"Write a Python function to check if a number is prime."
→ def is_prime(n): ...
→ Run against test cases to verify
```

### How Benchmarks Work

```
Benchmark Dataset         LLM              Scoring
┌─────────────┐      ┌─────────┐      ┌─────────────┐
│ Q1: ...     │      │         │      │             │
│ Q2: ...     │ ───→ │  Model  │ ───→ │  Compare    │
│ Q3: ...     │      │         │      │  to ground  │
│ ...         │      └─────────┘      │  truth      │
│ Q1000: ...  │                       └─────────────┘
└─────────────┘                              │
                                             ▼
                                       Score: 78.3%
```

### The Problem: Goodhart's Law Returns!

> "When a measure becomes a target, it ceases to be a good measure."

```
What we want:        "Build a model that's genuinely smart"
What we measure:     "Score on MMLU benchmark"
What happens:        Models optimize for MMLU specifically
Result:              High MMLU score ≠ actually smart
```

**Real examples of benchmark gaming:**

| Problem | What Happens |
|---------|--------------|
| Data contamination | Training data accidentally includes benchmark questions |
| Overfitting to format | Model learns benchmark patterns, not general skills |
| Cherry-picking | Companies report only benchmarks where they win |
| Benchmark saturation | Once models hit ~90%, benchmark stops being useful |

**Analogy:** Teaching to the test - memorizing SAT answers doesn't make you smarter.

---

## 3. Human Evaluation

Since benchmarks can be gamed, sometimes you need actual humans.

### How It Works

```
Human evaluator rates LLM responses on:

• Helpfulness    "Did it answer the question?"
• Accuracy       "Is the information correct?"
• Safety         "Is the response appropriate?"
• Clarity        "Is it easy to understand?"

Scale: 1-5 or thumbs up/down
```

### The Bias Problem ⚠️

**Critical point:** Human evaluation is subjective and prone to bias!

| Bias Type | Description | Example |
|-----------|-------------|---------|
| **Demographic bias** | Evaluators from different backgrounds have different preferences | Technical experts vs. general users |
| **Cultural bias** | What's "helpful" varies by culture | Direct vs. indirect communication styles |
| **Anchoring bias** | First responses influence judgment of later ones | First response seems better just because it's first |
| **Expertise bias** | Evaluators may lack domain knowledge | Non-expert can't judge medical accuracy |
| **Length bias** | Longer responses rated "better" even when not | Verbose ≠ correct |
| **Fluency bias** | Well-written wrong answers beat poorly-written correct ones | Style over substance |

**Mitigation strategies:**
- Use diverse evaluator pools
- Provide clear rubrics and training
- Blind evaluation (don't show model names)
- Multiple evaluators per response
- Statistical controls for known biases

### Pros and Cons

| Pros | Cons |
|------|------|
| Captures nuance | Expensive |
| Measures real helpfulness | Slow |
| Catches safety issues | Subjective / biased |
| Flexible to new tasks | Doesn't scale |

---

## 4. Online Evaluation: Real Users

### In-App Feedback

Every thumbs up/down provides evaluation data!

```
User: "Explain quantum computing"

Claude: "Quantum computing uses quantum mechanics..."

                                     [ 👍 ]  [ 👎 ]

This feedback → used to improve the model
```

**Pros:** Real tasks, real users, authentic signal
**Cons:** Selection bias (who clicks?), sparse data, no comparison

### Crowdsourcing: LMArena

The most trusted "apples-to-apples" LLM comparison.

**How it works:**

```
User submits prompt: "Where is the capital of France?"

┌─────────────────┐         ┌─────────────────┐
│    Model A      │         │    Model B      │
│   (hidden)      │         │   (hidden)      │
├─────────────────┤         ├─────────────────┤
│ "The capital of │         │ "Paris is the   │
│  France is      │         │  capital of     │
│  Paris..."      │         │  France..."     │
└─────────────────┘         └─────────────────┘

User picks:  [ A is better ]  [ B is better ]  [ Tie ]

After voting → Models revealed!
```

**Key insight:** Blind testing prevents brand bias!

### ELO Rating System

LMArena uses ELO ratings (same as chess) to rank models:

```
How ELO works:

If Model A (ELO 1500) beats Model B (ELO 1400):
→ A gains points, B loses points

If Model A (ELO 1500) beats Model B (ELO 1600):
→ A gains MORE points (upset victory!)
→ B loses MORE points

Over thousands of comparisons → stable ranking emerges
```

**Why ELO works well:**
- Handles transitive preferences (if A > B and B > C, then A > C)
- Self-correcting over time
- Relative ranking, not absolute scores
- Battle-tested system (used in chess for decades)

---

## Evaluation Methods Compared

| Method | Measures | Pros | Cons |
|--------|----------|------|------|
| **Perplexity** | Token prediction (fluency) | Fast, automatic | Doesn't measure usefulness |
| **Benchmarks** | Specific tasks | Standardized, comparable | Can be gamed, narrow |
| **Human evaluation** | Quality, safety, helpfulness | Captures nuance | Expensive, slow, biased |
| **User feedback** | Real-world satisfaction | Authentic | Sparse, selection bias |
| **LMArena** | Comparative preference | Blind, fair, scales | Limited to site visitors |

---

## Key Takeaways

1. **No single metric is enough** - real evaluation uses ALL methods together

2. **Perplexity ≠ usefulness** - fluent text can be wrong or unhelpful

3. **Benchmarks get gamed** - Goodhart's Law applies to evaluation too

4. **Human evaluation has bias** - evaluator background affects ratings

5. **Online evaluation is authentic but sparse** - real users, but limited data

6. **LMArena is currently the gold standard** - blind comparison prevents gaming

---

## Practical Recommendations

**For evaluating your own models:**

1. Start with perplexity (sanity check - is it generating fluent text?)
2. Run standard benchmarks (how does it compare to known models?)
3. Do human evaluation with diverse evaluators (is it actually useful?)
4. Deploy to small user group (how does it perform in the real world?)
5. Monitor user feedback continuously (is satisfaction stable over time?)

**For comparing models:**

1. Check LMArena leaderboard for overall ranking
2. Look at specific benchmarks for your use case (coding? math? general?)
3. Do your own human evaluation with your target users
4. Run A/B tests in production if possible

---

## Resources

- **LMArena Leaderboard:** https://lmarena.ai/
- **MMLU Benchmark:** Tests 57 subjects from STEM to humanities
- **HumanEval:** OpenAI's coding benchmark
- **GSM8K:** Grade school math problems
- **HellaSwag:** Common-sense completion task

---

*Teaching tip: When explaining evaluation, emphasize that it's an unsolved problem. Even the best evaluation methods have significant limitations. The field is actively researching better approaches.*
