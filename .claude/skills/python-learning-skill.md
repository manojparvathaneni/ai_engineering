# Python Learning Skill

> For creating in-depth Python learning materials separate from AI course content.

## Concept Folder Structure

```
python-learning/concepts/[concept-name]/
├── README.md       # Explanation with examples
├── examples.py     # Working code (inline deps)
├── exercises.py    # Practice problems
└── solutions.py    # Solutions (optional)
```

## README Template

```markdown
# [Concept Name]

> One-line description

## The Core Idea
[Plain English explanation of what this is]

## Why It Exists
[The problem it solves]

## How It Works
[Mechanics with simple examples]

## Progression
1. Basic usage
2. Intermediate patterns
3. Advanced usage

## Where You'll See It
| Library | Example | What It Does |
|---------|---------|--------------|
| ... | ... | ... |

## Key Insight
[The "aha" moment]

## Files
- examples.py - Working code
- exercises.py - Practice problems
```

## Examples File Template

```python
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
[Concept] Examples
==================

Run with: uv run examples.py

Progression:
1. [Basic]
2. [Intermediate]
3. [Advanced]
"""

# =============================================================================
# 1. BASIC
# =============================================================================

print("=" * 60)
print("1. [SECTION NAME]")
print("=" * 60)

# Code with comments explaining what's happening

print()

# ... more sections ...

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key takeaways:
1. ...
2. ...
""")
```

## Topics to Cover

### From AI Course (as encountered)
- Decorators (FastAPI routes, pytest fixtures)
- Async/await (API calls, streaming)
- Generators (data pipelines)
- Context managers (file handling, connections)
- Type hints (documentation, IDE support)

### Core Python
- `*args`, `**kwargs`
- List/dict comprehensions
- Lambda functions
- Property decorators
- Dunder methods

### Advanced
- Metaclasses
- Descriptors
- Abstract base classes
- Protocols (structural subtyping)

## Exercise Difficulty Levels

```python
# 🟢 Easy - Direct application of concept
# 🟡 Medium - Combine with other concepts
# 🔴 Hard - Edge cases, optimization, design
```

## Connection to AI Course

When a Python concept comes up in the AI course:
1. Note it in the concept's README
2. Show how it's used in AI context
3. Example: "Decorators are used in FastAPI's @app.get() for routing"
