# Teaching Materials Skill

## Guide Structure
1. Overview with key insight
2. Learning objectives
3. Core idea (plain English + analogy)
4. Sections with intuition → mechanics → code
5. Discussion questions
6. Exercises
7. Key takeaways

## Demo Structure (with uv inline dependencies)

All demos should be self-contained using PEP 723 inline dependencies:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.26.0",
#     "tiktoken>=0.5.0",
# ]
# ///
"""
[Topic Name] Demo
=================

Run with: uv run [script_name].py

Concepts covered:
1. [Concept 1]
2. [Concept 2]
"""

import numpy as np
import tiktoken

# Rest of demo...
```

**Key points:**
- Dependencies declared at top of file
- No separate requirements.txt needed
- Run with `uv run script.py`
- Comments explain "why"
- Print intermediate values
- ASCII diagrams when helpful

## Preferred Analogies
- Temperature: restaurant adventure level
- Top-k: only top k menu items
- Attention: "who should I pay attention to?"
- Pre-training: reading every book
- SFT: first week at job
- RLHF: performance reviews
