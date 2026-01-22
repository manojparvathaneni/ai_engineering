# Reinforcement Learning from Human Feedback (RLHF) Guide

## The Goal

Generate responses that are **preferred by humans**: correct, accurate, safe, helpful, polite.

---

## The Analogy: From Examples to Feedback

**SFT** = Showing a new employee example emails. They learn the format and can write reasonable emails.

**RLHF** = Having a manager review their drafts and say "this one's better than that one." Over time, they internalize what "good" means to this organization.

**The key shift:**
- SFT: "Here's what a good response looks like" (examples)
- RLHF: "This response is better than that one" (preferences)

---

## Why Preferences Matter

Imagine training a new barista:

**SFT approach:** Show them 1000 correctly-made drinks. They learn to make decent coffee.

**RLHF approach:** Have customers taste two drinks and pick the better one. The barista learns:
- "A little less syrup is preferred"
- "Latte art matters to some customers"  
- "Speed vs. quality tradeoffs"

These subtle preferences are **hard to demonstrate but easy to judge**. That's the power of RLHF.

---

## Two Types of Tasks

### Verifiable Tasks

**Examples:** Math problems, coding challenges, factual questions

**How to evaluate:** Computer can check - right or wrong

```
Prompt: What is 15 × 23?
Response A: 345 ✓
Response B: 355 ✗
```

No humans needed! The computer knows the answer.

### Unverifiable Tasks

**Examples:** Writing, brainstorming, explanations, advice

**How to evaluate:** Humans must judge - which is "better"?

```
Prompt: Write a poem about spring

Response A: Flowers bloom in morning light,
           Gentle breeze takes winter's flight...

Response B: Spring arrives with muddy boots,
           Tracking life through tender shoots...
```

Which is better? It depends on preference. Only humans can judge.

**Key insight:** RLHF is primarily needed for unverifiable tasks where "better" is subjective.

---

## The Three-Step RLHF Process

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Collect Preferences                                │
│  "Which response do you prefer?"                            │
│  → Dataset of (prompt, winner, loser) tuples                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Train Reward Model                                 │
│  Build an "automatic judge"                                 │
│  → Model that predicts human preferences                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: RL Fine-tuning                                     │
│  Optimize model to get high rewards                         │
│  → Model that generates preferred responses                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Collect Preference Data

### The Setup

1. Give a prompt to your SFT model
2. Generate two different responses (using sampling/temperature)
3. Show both to a human evaluator
4. Human picks which is better

```
Prompt: Explain why the sky is blue

Response A: The sky appears blue due to Rayleigh scattering, 
            a phenomenon where shorter wavelengths of light 
            are scattered more than longer wavelengths by 
            molecules in the atmosphere...

Response B: When sunlight hits our atmosphere, blue light 
            bounces around more than other colors because 
            it travels in smaller waves. It's like how small 
            balls bounce more in a pinball machine than big ones.

Human choice: B ✓ (clearer, good analogy)
```

### What You Get

A dataset of comparisons:
```python
preferences = [
    {"prompt": "Explain the sky...", "winner": "Response B", "loser": "Response A"},
    {"prompt": "Write a haiku...", "winner": "Response A", "loser": "Response B"},
    # ... 100K to 1M more comparisons
]
```

### The Wine Tasting Analogy

You don't need tasters to explain *why* they prefer Wine A over Wine B. You just need their choice. Patterns emerge from thousands of comparisons:
- Wines with balanced acidity score better
- Oak aging is preferred
- Too much sweetness loses

Similarly, patterns emerge from response comparisons:
- Direct answers beat evasive ones
- Accurate information beats confident-but-wrong
- Clear explanations beat jargon-heavy confusion

### Scale

| Company | Comparisons Collected |
|---------|----------------------|
| OpenAI (InstructGPT) | ~50,000 |
| Anthropic (Claude) | ~200,000+ |
| Open source (OpenAssistant) | ~100,000 |

---

## Step 2: Train the Reward Model

### The Goal

Build an **automatic judge** that predicts human preferences.

**The analogy:** After surveying 50,000 customers about coffee preferences, you build an AI that can predict "customers will prefer drink A over drink B" - without asking them. Now you have unlimited free judging!

### How It Works

The reward model takes (prompt, response) and outputs a score:

```python
reward_model("Explain gravity", "Gravity is a force that...") → 0.73
reward_model("Explain gravity", "Gravity happens when...")    → 0.61
```

Higher score = more likely to be preferred by humans.

### Architecture

Take a language model (often the SFT model itself), remove the vocabulary output head, add a single-number output:

```
┌─────────────────────────────────────────────┐
│  Input: prompt + response                   │
│  "Explain gravity" + "Gravity is a force.." │
│              ↓                              │
│  [Transformer layers - same as LLM]         │
│              ↓                              │
│  Final hidden state of last token           │
│              ↓                              │
│  Linear layer: hidden_size → 1              │
│              ↓                              │
│  Output: 0.73                               │
└─────────────────────────────────────────────┘
```

### Training Objective

Given a (winner, loser) pair: the winner should score higher.

```python
def reward_model_loss(prompt, winner, loser):
    score_winner = reward_model(prompt + winner)
    score_loser = reward_model(prompt + loser)
    
    # Bradley-Terry model
    # Loss is low when score_winner >> score_loser
    # Loss is high when they're close or reversed
    loss = -log(sigmoid(score_winner - score_loser))
    
    return loss
```

**The sports commentator analogy:** Train someone to predict match outcomes by showing them past matches with known winners. Over time, they learn to spot what makes a winner.

### What the Reward Model Learns

Through thousands of comparisons, patterns emerge:

| Humans Prefer | Over |
|---------------|------|
| Direct answers | Evasive non-answers |
| Accurate information | Confident but wrong |
| Appropriate length | Too short or verbose |
| Clear explanations | Jargon-heavy confusion |
| Admitting uncertainty | Hallucinating confidently |
| Safe responses | Harmful content |

---

## Step 3: RL Fine-tuning

### The Goal

Use the reward model to improve the language model.

**The analogy:** The barista now practices making drinks and gets instant feedback from the taste predictor. No need to wait for real customers - they can iterate thousands of times per day.

### The Training Loop

```python
for prompt in training_prompts:
    # 1. Generate response with current model
    response = model.generate(prompt)
    
    # 2. Score it with reward model
    reward = reward_model(prompt, response)
    
    # 3. Update model to increase reward
    rl_update(model, prompt, response, reward)
```

### Why "Reinforcement Learning"?

| RL Term | In RLHF |
|---------|---------|
| **Agent** | The language model |
| **Environment** | The conversation |
| **State** | Prompt + tokens so far |
| **Action** | Choosing the next token |
| **Reward** | Reward model score |
| **Policy** | Model's token probabilities |

**The dog training analogy:** 
- Dog tries an action
- Gets a treat if successful
- Learns to do more of what earns treats
- Eventually performs tricks reliably

The LLM is the dog. Good responses get treats (high reward). It learns to generate responses that earn treats.

### The Credit Assignment Problem

Reward comes at the END, but decisions happen at EVERY token:

```
Response: "The | capital | of | France | is | Paris | ."
                                                     ↑
                                             Reward given here

But which tokens contributed to the good score?
```

**The soccer team analogy:** Team wins 2-1. The reward (winning) comes at the end. But which passes, which defensive plays, which decisions led to the win? Credit assignment figures out who deserves the bonus.

### PPO: Proximal Policy Optimization

The standard RL algorithm used. Key features:

1. **Conservative updates:** Don't change too much at once
2. **Clipping:** If an update is too large, clip it back
3. **Stable training:** Prevents wild swings

**The driving lesson analogy:**
- Don't go from "never driven" to "race car driver" in one lesson
- Small, incremental improvements
- If student does something dangerous, instructor grabs wheel (clipping)

### The KL Penalty: Don't Drift Too Far

**The problem:** If you only optimize for reward, the model finds exploits.

**The barista analogy:** The taste predictor really likes foam. Solution? ALL FOAM, NO COFFEE. Scores high, but not what we wanted!

Real examples of reward hacking:
- Reward model likes confident responses → Model becomes overconfident
- Reward model likes detailed responses → Model adds meaningless padding
- Reward model likes safe responses → Model refuses everything

**The solution:** Keep the model close to the SFT version.

```python
total_reward = reward_score - β × KL_penalty
                    ↑                ↑
             what we want    penalty for drifting
```

**KL divergence** measures how different two distributions are. High KL = model drifted far from SFT.

**The analogy:** You want the barista to improve, but still make drinks recognizable as coffee. If they drift too far (serving smoothies because the predictor scores them high), pull them back.

---

## Alternative: DPO (Direct Preference Optimization)

### The Insight

What if we skip the reward model entirely?

```
RLHF: preferences → reward model → RL training → better model
DPO:  preferences ────────────────────────────→ better model
```

**The analogy:** Instead of building a taste predictor, just show the barista pairs of drinks: "customers preferred this one." They learn directly from comparisons.

### How DPO Works

Directly train the model so that:
- Winners become more likely
- Losers become less likely
- But not too far from original (implicit KL penalty)

```python
def dpo_loss(model, ref_model, prompt, winner, loser, beta):
    # How much does current model prefer winner over loser,
    # compared to reference model?
    
    log_ratio_w = model.log_prob(winner) - ref_model.log_prob(winner)
    log_ratio_l = model.log_prob(loser) - ref_model.log_prob(loser)
    
    # Winner's ratio should exceed loser's ratio
    return -log(sigmoid(beta * (log_ratio_w - log_ratio_l)))
```

### DPO vs RLHF

| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| **Complexity** | High (3 stages) | Low (standard fine-tuning) |
| **Stability** | Can be unstable | Standard loss function |
| **Compute** | RL + reward model | Just fine-tuning |
| **Performance** | Strong | Comparable |

**Current trend:** DPO is increasingly popular due to simplicity.

### GRPO (Group Relative Policy Optimization)

Another variant that:
- Groups multiple responses per prompt
- Computes relative rankings within group
- Often more stable than PPO

---

## Constitutional AI / RLAIF

### The Human Bottleneck

RLHF needs human labels. Humans are expensive, slow, and limited.

**The sommelier analogy:** Having professional sommeliers taste every wine pair. Accurate, but you can only do so many per day.

### AI as Judge

Use a capable AI to generate preferences instead:

```
Constitution (principles):
- Be helpful and accurate
- Don't assist with harmful activities
- Acknowledge uncertainty
- Be respectful

Process:
1. Generate response
2. Ask AI: "Does this follow the principles? How could it improve?"
3. AI generates improved response
4. Use (original, improved) as preference pair
5. Train with DPO
```

**The analogy:** Write a guidebook for good coffee. Train an AI to apply the guide. AI can judge unlimited coffee pairs. Humans just write the guide.

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Scalable** | AI generates millions of comparisons |
| **Consistent** | Same principles applied uniformly |
| **Transparent** | Principles are explicit |
| **Cheap** | No human labelers |

---

## Summary: The Complete Picture

```
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
```

## Key Takeaways

1. **Verifiable vs Unverifiable** - Determines if you need human judgment
2. **Preferences, not examples** - RLHF teaches comparisons SFT can't express
3. **Reward model = automatic judge** - Scales human preferences infinitely
4. **KL penalty prevents gaming** - Keep the model grounded
5. **DPO simplifies** - Skip reward model, optimize directly
6. **AI can judge AI** - Constitutional AI scales with principles

---

## Analogy Quick Reference

| Concept | Analogy |
|---------|---------|
| SFT | New employee shown example emails |
| RLHF | Manager reviewing drafts: "this one's better" |
| Reward model | Automated judge trained on human taste |
| PPO | Incremental driving lessons |
| KL penalty | Keep barista making coffee, not smoothies |
| Reward hacking | Teaching to the test |
| DPO | Learn from past judge decisions directly |
| Constitutional AI | Humans write law, AI applies at scale |
| Preference data | Wine tasting - just pick, don't explain |
| Credit assignment | Who on the team deserves the bonus? |
