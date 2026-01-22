# Decorators

> Functions that modify other functions. Used everywhere in Python (FastAPI, pytest, Flask, etc.)

## The Core Idea

A decorator is just a function that:
1. Takes a function as input
2. Returns a new function (usually wrapping the original)

```python
@decorator
def my_function():
    pass

# Is exactly the same as:
def my_function():
    pass
my_function = decorator(my_function)
```

## Why They Exist

**Without decorators:**
```python
def log_calls(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

def add(a, b):
    return a + b

add = log_calls(add)  # Manual wrapping - ugly!
```

**With decorators:**
```python
@log_calls
def add(a, b):
    return a + b
# Clean and readable!
```

## Progression

1. **Simple decorator** - No arguments to decorator
2. **Decorator with arguments** - `@decorator(arg)`
3. **Class decorator** - Using a class instead of function
4. **Stacking decorators** - Multiple decorators on one function

## Where You'll See Them

| Library | Example | What It Does |
|---------|---------|--------------|
| FastAPI | `@app.get("/")` | Route registration |
| pytest | `@pytest.fixture` | Test setup |
| functools | `@lru_cache` | Memoization |
| dataclasses | `@dataclass` | Auto-generate methods |
| Flask | `@app.route()` | Route registration |

## Key Insight

Decorators are Python's way of doing **aspect-oriented programming** - adding behavior (logging, caching, auth) without modifying the core function.

## Files

- `examples.py` - Working code examples
- `exercises.py` - Practice problems
- `solutions.py` - Solutions
