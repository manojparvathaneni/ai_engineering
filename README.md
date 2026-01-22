# AI Engineering Course - Learning Repository

> A comprehensive learning repository for the AI Engineering cohort (Jan 17 - Feb 22, 2026), combining course projects, teaching materials, research, and hands-on exploration.

## Course Overview

**Program:** AI Engineering Cohort 3 (Circle)  
**Duration:** 6 weeks (Jan 17 - Feb 22, 2026)  
**Schedule:**
- Live Build Session: Saturdays 10-11:30 AM PT
- Office Hours: Wednesdays 5-6 PM PT
- Guided Learning: 2-3 hours/week
- Project Work: 1-3 hours/week

**Course Repository:** https://github.com/bytebyteai/ai-engineering-cohort-3

## Repository Structure

```
ai-engineering/
├── README.md                 # This file
├── CLAUDE.md                 # Context for Claude Code
├── .cursorrules              # IDE integration
│
├── week-01-llm-foundations/  # Course weeks...
├── week-02-rag-prompting/
├── week-03-agents-tools/
├── week-04-reasoning-research/
├── week-05-multimodal/
│
├── python-learning/          # Python deep dives (separate from AI)
│   ├── concepts/             # Decorators, generators, async, etc.
│   ├── exercises/            # Practice problems
│   ├── projects/             # Small Python projects
│   ├── scripts/              # Utility scripts
│   └── references/           # Cheat sheets
│
├── research/                 # Papers, notes
├── teaching-sessions/        # Session prep
└── shared/                   # Common utilities
```

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <your-repo-url>
cd ai-engineering

# Run a teaching demo (inline deps, no setup needed!)
uv run week-01-llm-foundations/teaching-materials/demos/tokenization_demo.py

# For a project
cd week-01-llm-foundations/project-01-llm-playground
uv sync                    # Install dependencies
uv run python main.py      # Run the project
```

## Projects Overview

| Week | Project | Status |
|------|---------|--------|
| 1 | LLM Playground | 🔨 In Progress |
| 2 | Customer Support Chatbot | ⏳ Upcoming |
| 3 | Ask-the-Web Agent | ⏳ Upcoming |
| 4 | Deep Research | ⏳ Upcoming |
| 5 | Multimodal Agent | ⏳ Upcoming |
| 6 | Capstone | ⏳ Upcoming |

## License

Personal learning repository. Course materials © their respective owners.
