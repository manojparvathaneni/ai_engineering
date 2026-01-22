# Python Quick Reference

> Common patterns and syntax quick lookups

## Functions

```python
# Basic
def func(a, b):
    return a + b

# Default arguments
def func(a, b=10):
    return a + b

# *args (variable positional)
def func(*args):
    for arg in args:
        print(arg)

# **kwargs (variable keyword)
def func(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Both
def func(*args, **kwargs):
    pass

# Type hints
def func(name: str, age: int = 0) -> str:
    return f"{name} is {age}"
```

## List Comprehensions

```python
# Basic
[x * 2 for x in range(10)]

# With condition
[x for x in range(10) if x % 2 == 0]

# Nested
[[i * j for j in range(3)] for i in range(3)]

# Dict comprehension
{k: v * 2 for k, v in d.items()}

# Set comprehension
{x % 3 for x in range(10)}
```

## Lambda Functions

```python
# Basic
f = lambda x: x * 2

# Multiple args
f = lambda x, y: x + y

# With sorted
sorted(items, key=lambda x: x.name)

# With map/filter
list(map(lambda x: x * 2, [1, 2, 3]))
list(filter(lambda x: x > 0, [-1, 0, 1, 2]))
```

## Context Managers

```python
# File handling
with open("file.txt", "r") as f:
    content = f.read()

# Custom (class)
class MyContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Custom (decorator)
from contextlib import contextmanager

@contextmanager
def my_context():
    # setup
    yield value
    # teardown
```

## Decorators

```python
# Basic
def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@decorator
def my_func():
    pass

# With arguments
def decorator(arg):
    def actual_decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return actual_decorator

@decorator(arg="value")
def my_func():
    pass
```

## Classes

```python
# Basic
class MyClass:
    class_var = "shared"
    
    def __init__(self, value):
        self.instance_var = value
    
    def method(self):
        return self.instance_var
    
    @classmethod
    def from_string(cls, s):
        return cls(int(s))
    
    @staticmethod
    def utility():
        return "no self needed"
    
    @property
    def computed(self):
        return self.instance_var * 2

# Dataclass
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    label: str = "origin"
```

## Async/Await

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return "data"

async def main():
    result = await fetch_data()
    
    # Multiple concurrent
    results = await asyncio.gather(
        fetch_data(),
        fetch_data(),
    )

asyncio.run(main())
```

## Error Handling

```python
try:
    risky_operation()
except ValueError as e:
    print(f"Value error: {e}")
except (TypeError, KeyError):
    print("Type or key error")
except Exception as e:
    print(f"Unexpected: {e}")
else:
    print("No error occurred")
finally:
    print("Always runs")

# Raising
raise ValueError("message")

# Custom exception
class MyError(Exception):
    pass
```

## Common Patterns

```python
# Ternary
value = a if condition else b

# Walrus operator (:=)
if (n := len(items)) > 10:
    print(f"Too many: {n}")

# Unpacking
a, b, *rest = [1, 2, 3, 4, 5]
first, *middle, last = items

# Dictionary merge (3.9+)
merged = dict1 | dict2

# f-strings
f"{value:.2f}"      # 2 decimal places
f"{value:>10}"      # right align, 10 chars
f"{value!r}"        # repr
f"{obj.attr=}"      # debug (shows "obj.attr=value")
```

---

*Quick reference - see concepts/ for deep dives*
