# CLAUDE.md - AI Engineering Learning Repository

> Context for Claude Code to assist with this repository effectively.

## About This Repository

This is Manoj's learning repository for the **AI Engineering Cohort 3** course (Jan 17 - Feb 22, 2026).

Purposes:
1. **Course Projects**: 5 projects + capstone across 6 weeks
2. **Teaching Preparation**: Materials for AI/Python sessions (starting Feb 1st)
3. **Research & Exploration**: Papers, notes, experiments
4. **Learning Documentation**: Progress tracking

## Owner Context

**Learning Style**:
- ~30 minute content chunks
- Deep dives, asks "why" questions
- Creates teaching materials from learning
- Intuition before mechanics
- Likes analogies (temperature = "restaurant adventure level")
- Minimal math background but building it

## Course Structure

- Week 1: LLM Foundations → Project 1: LLM Playground ✅
- Week 2: RAG & Prompting → Project 2: Customer Support Chatbot
- Week 3: Agents & Tools → Project 3: Ask-the-Web Agent
- Week 4: Reasoning → Project 4: Deep Research
- Week 5: Multimodal → Project 5: Multimodal Agent
- Week 6: Capstone (separate repo)

## Python Learning (Separate from AI Course)

The `python-learning/` folder is for in-depth Python exploration:
- **concepts/** - Deep dives (decorators, generators, async, etc.)
- **exercises/** - Practice problems
- **projects/** - Small Python projects
- **scripts/** - Utility scripts, experiments
- **references/** - Cheat sheets

When Manoj encounters Python concepts in the AI course that need deeper understanding, they get documented here with examples and exercises.

## Week 1 Progress: ✅ COMPLETE

**Teaching Materials Created:**
- 13 guides (llm-intro through system-design)
- 13 demos (crawler through system_design)
- Including: pretraining_simulation.py (368-param working LM!)

**Key Concepts Covered:**
- Pre-training: Crawling, Cleaning, Tokenization, Architecture, Training, Generation
- Post-training: SFT, RLHF, Evaluation, System Design

## Package Management: uv

**We use `uv` for all Python package management.**

### For Scripts (Inline Dependencies - PEP 723)
```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "tiktoken"]
# ///
```
Run with: `uv run script.py`

### For Projects (pyproject.toml)
```bash
cd project-folder/
uv run python main.py      # Runs with project deps
uv run pytest              # Runs tests
uv add package-name        # Add dependency
uv sync                    # Install all deps
```

### Key uv Commands
```bash
uv run script.py           # Run script with inline deps
uv run --with pkg script.py # Run with extra package
uv init                    # Create new project
uv add package             # Add to pyproject.toml
uv sync                    # Sync environment
uv venv                    # Create .venv if needed
```

## Project Conventions

- **Scripts**: Use inline dependencies (PEP 723 format)
- **Projects**: Use pyproject.toml
- Python 3.10+, type hints, docstrings
- Use pathlib for paths, python-dotenv for secrets
- Cross-platform: WSL, Linux, Mac
- NO requirements.txt files (use uv instead)

## Key Analogies

| Concept | Analogy |
|---------|---------|
| Temperature | Restaurant adventure level |
| Top-k | Only top k menu items |
| Attention | "Who should I pay attention to?" |
| Pre-training | Reading every book in library |
| SFT | First week at new job |
| RLHF | Performance reviews |

## Communication Style

- Be direct and practical
- Explain the "why"
- Use analogies for complex concepts
- Provide working code
- Connect to what's been learned
- Suggest teaching angles
