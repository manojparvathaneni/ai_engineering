# Python Learning

> In-depth Python exploration, separate from AI course content.

## Purpose

As I learn Python more deeply through the AI Engineering course, this folder captures:
- Deep dives on Python concepts
- Practice exercises and challenges
- Small projects for skill building
- Utility scripts and experiments
- Quick references and cheat sheets

## Structure

```
python-learning/
├── concepts/           # Deep dives on specific topics
│   ├── decorators/
│   ├── generators/
│   ├── async-await/
│   ├── context-managers/
│   ├── metaclasses/
│   └── ...
├── exercises/          # Practice problems by topic
│   ├── data-structures/
│   ├── algorithms/
│   └── ...
├── projects/           # Small Python projects
├── scripts/            # Utility scripts, one-offs
└── references/         # Cheat sheets, quick refs
```

## Topics to Explore

### Core Python
- [ ] Decorators (function, class, with arguments)
- [ ] Generators and iterators
- [ ] Context managers (`with` statement)
- [ ] `*args` and `**kwargs` patterns
- [ ] Type hints and typing module
- [ ] Dataclasses and attrs

### Intermediate
- [ ] Async/await and asyncio
- [ ] Metaclasses
- [ ] Descriptors
- [ ] Abstract base classes
- [ ] Multiple inheritance and MRO

### Practical Patterns
- [ ] Factory patterns
- [ ] Dependency injection
- [ ] Configuration management
- [ ] Error handling patterns
- [ ] Logging best practices

### Performance
- [ ] Profiling (cProfile, line_profiler)
- [ ] Memory optimization
- [ ] Cython basics
- [ ] Multiprocessing vs threading

### Testing
- [ ] pytest fixtures and parametrize
- [ ] Mocking and patching
- [ ] Property-based testing (hypothesis)
- [ ] Test organization patterns

## Concepts Format

Each concept folder contains:
```
concepts/decorators/
├── README.md           # Explanation with examples
├── examples.py         # Working code examples
├── exercises.py        # Practice problems
└── solutions.py        # Solutions (hidden until needed)
```

## Running Scripts

All scripts use uv with inline dependencies:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
```

Run with: `uv run script.py`

## Connection to AI Course

Python concepts I encounter in the AI course that need deeper exploration get documented here:
- Decorators → Used heavily in FastAPI
- Async/await → API calls, streaming
- Generators → Data pipelines, memory efficiency
- Type hints → Better code documentation

---

*Learning Python in depth alongside AI Engineering*
