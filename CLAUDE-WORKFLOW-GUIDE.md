# Claude Code Workflow Guide

> A practical guide for using Claude Code with your AI Engineering course, including prompts, examples, and best practices for coordinating between Claude.ai and Claude Code.

---

## Table of Contents

1. [When to Use What](#when-to-use-what)
2. [Getting Started with Claude Code](#getting-started)
3. [Task-Specific Prompts with Examples](#task-specific-prompts)
4. [Workflow Patterns](#workflow-patterns)
5. [Handoff Between Claude.ai and Claude Code](#handoff-patterns)
6. [Tips and Best Practices](#tips-and-best-practices)

---

## When to Use What

| Task | Use Claude.ai | Use Claude Code |
|------|---------------|-----------------|
| **Learning/discussing concepts** | ✅ Best | ❌ |
| **Brainstorming ideas** | ✅ Best | ❌ |
| **Creating files/code** | ✅ Can do | ✅ Best (in your actual repo) |
| **Debugging code** | ✅ Good | ✅ Best (can run/test) |
| **Exploring your codebase** | ❌ | ✅ Best |
| **Running commands** | ❌ | ✅ Only option |
| **Git operations** | ❌ | ✅ Only option |
| **Installing packages** | ❌ | ✅ Only option |
| **Multi-file refactoring** | ❌ | ✅ Best |
| **Research/web search** | ✅ Best | ❌ Limited |

### The Sweet Spot

```
Claude.ai                          Claude Code
─────────────────────────────────────────────────────────
Learn concepts ──────────────────► Implement in code
Brainstorm approach ─────────────► Build the feature
Discuss architecture ────────────► Create the files
Debug logic (discuss) ───────────► Fix and test
Review teaching materials ───────► Generate final files
```

---

## Getting Started with Claude Code

### First Time Setup

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to your project
cd ~/projects/ai-engineering

# Start Claude Code
claude
```

### Initial Context Prompt

When you first start Claude Code on this project, use:

```
I'm working on my AI Engineering course repository. 

Please read CLAUDE.md for full context about:
- Course structure (6 weeks, 5 projects + capstone)  
- My learning style and teaching goals
- Current progress (Week 1 complete)
- Project conventions

The repo structure is already set up. What would you like help with?
```

Claude Code will then have context for all future interactions in that session.

---

## Task-Specific Prompts with Examples

### 1. Starting a New Week

**When:** Beginning of each course week

**Prompt Template:**
```
I'm starting Week [X]: [Topic Name] of my AI Engineering course.

Please help me:
1. Review the week folder structure (already created)
2. Update the README with detailed topics from syllabus
3. Set up the project with a virtual environment
4. Create any starter files needed

Topics for this week:
[Paste syllabus topics here]
```

**Example - Starting Week 2:**
```
I'm starting Week 2: RAG & Prompt Engineering of my AI Engineering course.

Please help me:
1. Review the week folder structure (already created)
2. Update the README with detailed topics from syllabus
3. Set up the project with a virtual environment
4. Create any starter files needed

Topics for this week:
- Fine-tuning: PEFT, Adapters, LoRA
- Prompt Engineering: Few-shot, zero-shot, CoT, role prompting
- RAG Overview
- Retrieval: Document parsing, chunking, indexing, embeddings
- Generation: Search methods, prompt engineering for RAG
- RAFT training technique
- Evaluation: context relevance, faithfulness, answer correctness
```

**What Claude Code will do:**
- Check `week-02-rag-prompting/` structure
- Update README.md with detailed topic breakdown
- Run `python -m venv venv` in project folder
- Create `requirements.txt` with relevant packages
- Maybe create starter notebooks or config files

---

### 2. Creating Teaching Materials

**When:** After learning a topic, want to create guide + demo

**Prompt Template:**
```
I've finished learning about [TOPIC] from the course. Let's create teaching materials.

Key concepts I learned:
- [Concept 1]
- [Concept 2]
- [Concept 3]

Key insights/analogies I want to include:
- [Insight 1]
- [Insight 2]

Please create:
1. A teaching guide (markdown) in week-XX/teaching-materials/guides/
2. A runnable demo (python) in week-XX/teaching-materials/demos/

Follow my teaching style from the skills file (intuition first, analogies, practical examples).
```

**Example - Creating RAG Materials:**
```
I've finished learning about RAG retrieval from the course. Let's create teaching materials.

Key concepts I learned:
- Document parsing (rule-based vs AI-based)
- Chunking strategies (fixed size, semantic, recursive)
- Indexing types (keyword, vector, hybrid)
- Embedding models and how they work

Key insights/analogies I want to include:
- Chunking is like "how do you break a book into index cards?"
- Embeddings are "coordinates on a meaning map"
- Vector search is "finding nearby points in meaning space"

Please create:
1. A teaching guide in week-02-rag-prompting/teaching-materials/guides/
2. A runnable demo in week-02-rag-prompting/teaching-materials/demos/

Follow my teaching style from the skills file.
```

---

### 3. Building Project Features

**When:** Working on course projects

**Prompt Template:**
```
I'm working on Project [X]: [Name].

Current status:
[What exists, what's working]

I need to implement:
[Feature description]

Requirements:
- [Requirement 1]
- [Requirement 2]

Please help me build this. Start by reviewing the existing code in [path].
```

**Example - Adding RAG to Chatbot:**
```
I'm working on Project 2: Customer Support Chatbot.

Current status:
- Basic FastAPI backend exists in week-02/project-02/backend/
- Can accept messages and return responses
- No retrieval yet - just passes to LLM directly

I need to implement:
- Vector store for knowledge base documents
- Retrieval step before LLM call
- Context injection into prompt

Requirements:
- Use ChromaDB for vector store
- Use sentence-transformers for embeddings
- Retrieved context should be added to system prompt

Please help me build this. Start by reviewing the existing code in week-02-rag-prompting/project-02-customer-support-chatbot/
```

---

### 4. Debugging Issues

**When:** Something isn't working

**Prompt Template:**
```
I'm getting an error in [project/file].

Error message:
```
[paste full error]
```

What I was trying to do:
[Description]

Relevant code is in: [path]

Please investigate and fix this.
```

**Example:**
```
I'm getting an error in the RAG chatbot.

Error message:
```
chromadb.errors.InvalidCollectionException: Collection 'knowledge_base' does not exist.
```

What I was trying to do:
Query the vector store after adding documents

Relevant code is in: week-02-rag-prompting/project-02-customer-support-chatbot/backend/

Please investigate and fix this.
```

---

### 5. Research and Paper Notes

**When:** Found a paper or concept to explore

**Prompt Template:**
```
I encountered [paper/concept] and want to document it.

What I know so far:
[Brief description]

Please help me:
1. Create a notes file in research/papers/[name]/ or research/notes/concepts/
2. Use the research template from .claude/skills/research-skill.md
3. Connect it to what I'm learning in Week [X]
```

**Example:**
```
I encountered the RAFT paper (Retrieval Augmented Fine Tuning) and want to document it.

What I know so far:
- It's a technique to make models better at RAG
- Combines fine-tuning with retrieval training
- Mentioned in Week 2 syllabus

Please help me:
1. Create a notes file in research/papers/raft/
2. Use the research template from .claude/skills/research-skill.md
3. Connect it to what I'm learning in Week 2
```

---

### 6. End of Week Wrap-up

**When:** Finishing a week of the course

**Prompt Template:**
```
I've completed Week [X]: [Topic].

Please help me:
1. Summarize what was covered (check teaching-materials/)
2. List all materials created this week
3. Update the week README with final status
4. Note any gaps or TODO items
5. Check that all demos run correctly
```

**Example:**
```
I've completed Week 2: RAG & Prompt Engineering.

Please help me:
1. Summarize what was covered (check teaching-materials/)
2. List all materials created this week
3. Update the week README with final status
4. Note any gaps or TODO items
5. Check that all demos run correctly
```

---

### 7. Preparing for Teaching

**When:** Getting ready for your Tuesday/Thursday sessions

**Prompt Template:**
```
I'm teaching [topic] on [day]. 

Materials location: week-XX/teaching-materials/

Please help me:
1. Create a session outline (X minutes)
2. Identify the 3-5 key points to emphasize
3. Prepare simple examples I can use on whiteboard
4. List potential student questions and answers
5. Create a one-page cheat sheet I can reference
```

**Example:**
```
I'm teaching "Introduction to RAG" on Tuesday.

Materials location: week-02-rag-prompting/teaching-materials/

Please help me:
1. Create a session outline (45 minutes)
2. Identify the 3-5 key points to emphasize
3. Prepare simple examples I can use on whiteboard
4. List potential student questions and answers
5. Create a one-page cheat sheet I can reference

Save the session plan in teaching-sessions/session-notes/
```

---

### 8. Quick Tasks

**Running a demo:**
```
Run the tokenization demo: uv run week-01/teaching-materials/demos/tokenization_example.py
```

**Running tests:**
```
Run tests in week-02/project-02/: cd there, then uv run pytest
```

**Setting up a project:**
```
Initialize the project in week-02/project-02/ with uv init, add fastapi and anthropic
```

**Adding dependencies:**
```
Add chromadb to the project: uv add chromadb
```

**Git operations:**
```
Create a commit with all changes from today's work on Project 2.
Message: "Add vector store and retrieval to chatbot"
```

**Code review:**
```
Review the code in week-02/project-02/backend/retrieval.py
Check for:
- Error handling
- Edge cases
- Code clarity
Suggest improvements.
```

---

## Workflow Patterns

### Pattern 1: Learn → Discuss → Build

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. LEARN (Course)                                               │
│    Watch video / read material                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. DISCUSS (Claude.ai)                                          │
│    "I just learned about X. Let's go deeper..."                 │
│    - Ask questions                                              │
│    - Explore edge cases                                         │
│    - Develop analogies                                          │
│    - Plan teaching approach                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. BUILD (Claude Code)                                          │
│    "Create teaching materials for X..."                         │
│    - Generate guide                                             │
│    - Create demo                                                │
│    - Test code runs                                             │
│    - Commit to repo                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern 2: Design → Implement → Debug

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DESIGN (Claude.ai)                                           │
│    "I need to build a feature that does X..."                   │
│    - Discuss architecture                                       │
│    - Explore options                                            │
│    - Decide on approach                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. IMPLEMENT (Claude Code)                                      │
│    "Build the X feature following this design..."               │
│    - Create files                                               │
│    - Write code                                                 │
│    - Run tests                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. DEBUG (Either)                                               │
│    Claude.ai: "Why might this error happen?"                    │
│    Claude Code: "Fix this error: [paste]"                       │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern 3: Python Deep Dive (AI Course → Python Learning)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. ENCOUNTER (AI Course)                                        │
│    See Python concept (e.g., decorators in FastAPI)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. UNDERSTAND (Claude.ai)                                       │
│    "I saw @app.get() in FastAPI. How do decorators work?"       │
│    - Get explanation                                            │
│    - Ask follow-up questions                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. DOCUMENT (Claude Code)                                       │
│    "Create a deep dive on decorators in python-learning/"       │
│    - README with explanation                                    │
│    - examples.py with working code                              │
│    - exercises.py for practice                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern 4: Research → Document → Apply

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. ENCOUNTER (Course/Reading)                                   │
│    Find interesting paper or concept                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. UNDERSTAND (Claude.ai)                                       │
│    "Explain the key ideas in the RAFT paper..."                 │
│    - Summarize                                                  │
│    - Connect to course                                          │
│    - Discuss implications                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. DOCUMENT (Claude Code)                                       │
│    "Create research notes for RAFT..."                          │
│    - Create notes file                                          │
│    - Add to paper index                                         │
│    - Link to relevant week                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Handoff Between Claude.ai and Claude Code

### Scenario 1: You discussed with me, now want Claude Code to build it

**In Claude.ai (me), ask:**
```
Create a handoff summary for Claude Code to implement what we discussed.
Include:
- What we decided
- Key requirements
- Any code snippets or structures we designed
- What files need to be created/modified
```

**I'll generate something like:**
```
## Handoff to Claude Code

### Task
Implement RAG retrieval for the customer support chatbot

### Decisions Made
- Use ChromaDB for vector store
- Use all-MiniLM-L6-v2 for embeddings
- Chunk size: 500 chars with 50 char overlap
- Retrieve top 3 chunks per query

### Files to Create/Modify
1. `backend/retrieval.py` - New file for retrieval logic
2. `backend/embeddings.py` - New file for embedding functions
3. `backend/main.py` - Modify to add retrieval step
4. `requirements.txt` - Add chromadb, sentence-transformers

### Code Structure
[Any pseudocode or structure we discussed]
```

**Then paste that into Claude Code.**

---

### Scenario 2: Claude Code built something, want my review

**In Claude Code:**
```
Export a summary of what was created today for review in Claude.ai.
Include:
- Files created/modified
- Key decisions made
- Any questions or concerns
```

**Then paste that summary into Claude.ai:**
```
Claude Code just built [feature]. Here's the summary:

[Paste summary]

Please review:
1. Does the approach make sense?
2. Any edge cases we missed?
3. How should I test this?
4. What would you teach differently?
```

---

### Scenario 3: Hit a conceptual blocker in Claude Code

**When Claude Code is stuck on "why" questions:**

```
I'm working with Claude Code on [feature] and hit a conceptual question:

[Question]

Current context:
[What we're building]

Can you help me understand this so I can guide the implementation?
```

---

### Scenario 4: Need to sync context between sessions

**Starting new Claude Code session after discussing with me:**

```
I was discussing [topic] with Claude.ai. Here's the context:

[Paste relevant parts of our conversation or summary]

Please continue from here and help me implement this.
```

**Starting new Claude.ai session after working with Claude Code:**

```
I was working with Claude Code on [project]. Here's what was built:

[Paste summary of what was created]

I have questions about:
1. [Question]
2. [Question]
```

---

## Tips and Best Practices

### For Claude Code

1. **Always reference CLAUDE.md first**
   ```
   Please read CLAUDE.md for context before starting.
   ```

2. **Be specific about file locations**
   ```
   Create the file in week-02-rag-prompting/teaching-materials/guides/
   ```

3. **Ask for verification**
   ```
   After creating, run the demo to make sure it works.
   ```

4. **Use incremental commits**
   ```
   Commit this change before moving to the next feature.
   ```

5. **Check before overwriting**
   ```
   Review what's in that file first, then update it.
   ```

### For Claude.ai (Me)

1. **Share what you're learning**
   ```
   I just learned X. Here's what I understood: [summary]
   What am I missing? What would you add?
   ```

2. **Ask for teaching angles**
   ```
   How would you explain this to someone who doesn't know calculus?
   ```

3. **Request handoff summaries**
   ```
   Summarize this discussion so I can hand it off to Claude Code.
   ```

4. **Bring back questions from implementation**
   ```
   While implementing, I realized I don't understand Y. Can we discuss?
   ```

### General

1. **Keep a scratch file for handoffs**
   Create `shared/handoff-notes.md` to track context between tools

2. **Regular syncs**
   At end of each work session, document what was done

3. **Use the prompts file**
   `CLAUDE-CODE-PROMPTS.md` has templates - copy and customize

4. **Trust the skills**
   The `.claude/skills/` files give Claude Code your preferences

---

## Quick Reference Card

### Starting Tasks
| Task | Tool | Prompt Start |
|------|------|--------------|
| Learn concept | Claude.ai | "I'm learning about X..." |
| Build feature | Claude Code | "Build X in [path]..." |
| Create materials | Claude Code | "Create teaching guide for X..." |
| Debug issue | Either | "Getting error: [error]..." |
| Research topic | Claude.ai | "Explain [paper/concept]..." |

### Common Commands for Claude Code
```
# Setup
"Read CLAUDE.md for context"
"Initialize project with uv init in [path]"
"Add [packages] to the project with uv add"

# Running
"Run [demo.py] with uv run"
"Start the backend with uv run uvicorn main:app --reload"
"Run tests with uv run pytest"

# Building
"Create [file] in [path]"
"Add [feature] to [existing file]"
"Run the tests and fix failures"

# Maintenance
"Update README with current status"
"Commit changes with message: [msg]"
"Check all demos in week-X run correctly"
```

### Handoff Phrases
```
# Me → Claude Code
"Create a handoff summary for Claude Code"

# Claude Code → Me
"I have a conceptual question about [topic]"

# Sync context
"Here's what was discussed/built: [summary]"
```

---

*This guide lives at: `ai-engineering/CLAUDE-WORKFLOW-GUIDE.md`*
