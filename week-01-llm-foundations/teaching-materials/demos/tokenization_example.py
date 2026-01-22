# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "tiktoken>=0.5.0",
# ]
# ///
"""
Tokenization Demo (Example Format)
==================================

Run with: uv run tokenization_example.py

This demonstrates:
1. How inline dependencies work with uv
2. Basic tiktoken usage
3. The format for all teaching demos
"""

import tiktoken


def main():
    print("=" * 60)
    print("TOKENIZATION DEMO")
    print("=" * 60)
    
    # Load the GPT-4 tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    
    print(f"\nVocabulary size: {enc.n_vocab:,} tokens")
    
    # Example tokenization
    examples = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "ChatGPT",
        "def hello(): print('hi')",
    ]
    
    print("\n" + "-" * 60)
    print("TOKENIZATION EXAMPLES")
    print("-" * 60)
    
    for text in examples:
        tokens = enc.encode(text)
        token_strs = [enc.decode([t]) for t in tokens]
        
        print(f"\nText: \"{text}\"")
        print(f"  Tokens: {len(tokens)}")
        print(f"  Token strings: {token_strs}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
    LLMs never see words - only token IDs!
    
    "Hello" → [9906] → embedding → transformer → ...
    
    This is why LLMs struggle with character-level tasks
    like counting letters in "strawberry".
    """)


if __name__ == "__main__":
    main()
