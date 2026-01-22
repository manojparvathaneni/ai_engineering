# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Decorators Examples
===================

Run with: uv run examples.py

Progression:
1. Simple decorator
2. Decorator that preserves metadata
3. Decorator with arguments
4. Class-based decorator
5. Real-world examples
"""

import functools
import time


# =============================================================================
# 1. SIMPLE DECORATOR
# =============================================================================

print("=" * 60)
print("1. SIMPLE DECORATOR")
print("=" * 60)

def simple_logger(func):
    """A basic decorator that logs when a function is called."""
    def wrapper(*args, **kwargs):
        print(f"  Calling: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"  Finished: {func.__name__}")
        return result
    return wrapper

@simple_logger
def greet(name):
    """Greet someone."""
    print(f"  Hello, {name}!")

greet("World")
print()


# =============================================================================
# 2. PRESERVING METADATA WITH @functools.wraps
# =============================================================================

print("=" * 60)
print("2. PRESERVING METADATA")
print("=" * 60)

def better_logger(func):
    """Decorator that preserves the original function's metadata."""
    @functools.wraps(func)  # This preserves __name__, __doc__, etc.
    def wrapper(*args, **kwargs):
        print(f"  Calling: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@better_logger
def documented_function():
    """This docstring should be preserved."""
    pass

print(f"  Function name: {documented_function.__name__}")
print(f"  Docstring: {documented_function.__doc__}")
print()


# =============================================================================
# 3. DECORATOR WITH ARGUMENTS
# =============================================================================

print("=" * 60)
print("3. DECORATOR WITH ARGUMENTS")
print("=" * 60)

def repeat(times):
    """Decorator that repeats a function call N times."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(times):
                print(f"  Call {i + 1}/{times}")
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def say_hello():
    print("  Hello!")

say_hello()
print()


# =============================================================================
# 4. TIMING DECORATOR (Practical Example)
# =============================================================================

print("=" * 60)
print("4. TIMING DECORATOR")
print("=" * 60)

def timer(func):
    """Measure execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"  {func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    """Simulate a slow operation."""
    time.sleep(0.1)
    return "done"

slow_function()
print()


# =============================================================================
# 5. STACKING DECORATORS
# =============================================================================

print("=" * 60)
print("5. STACKING DECORATORS")
print("=" * 60)

@timer
@repeat(times=2)
def stacked_example():
    print("  Inside function")

print("  Note: Decorators apply bottom-up (@repeat first, then @timer)")
stacked_example()
print()


# =============================================================================
# 6. CLASS-BASED DECORATOR
# =============================================================================

print("=" * 60)
print("6. CLASS-BASED DECORATOR")
print("=" * 60)

class CountCalls:
    """Decorator that counts how many times a function is called."""
    
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"  Call #{self.count}")
        return self.func(*args, **kwargs)

@CountCalls
def counted_function():
    print("  Doing something")

counted_function()
counted_function()
counted_function()
print(f"  Total calls: {counted_function.count}")
print()


# =============================================================================
# 7. REAL-WORLD PATTERN: Retry Decorator
# =============================================================================

print("=" * 60)
print("7. RETRY DECORATOR (Real-World Pattern)")
print("=" * 60)

def retry(max_attempts=3, delay=0.1):
    """Retry a function on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"  Attempt {attempt} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

# Simulating a flaky function
call_count = 0

@retry(max_attempts=3)
def flaky_function():
    global call_count
    call_count += 1
    if call_count < 3:
        raise ConnectionError("Network error")
    return "Success!"

result = flaky_function()
print(f"  Result: {result}")
print()


# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key patterns:

1. Basic structure:
   def decorator(func):
       @functools.wraps(func)
       def wrapper(*args, **kwargs):
           # before
           result = func(*args, **kwargs)
           # after
           return result
       return wrapper

2. With arguments (extra layer):
   def decorator(arg):
       def actual_decorator(func):
           @functools.wraps(func)
           def wrapper(*args, **kwargs):
               # use arg here
               return func(*args, **kwargs)
           return wrapper
       return actual_decorator

3. Always use @functools.wraps to preserve metadata!

4. Decorators apply bottom-up when stacked.
""")
