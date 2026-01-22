"""
Tokenization for LLMs: Teaching Demonstration
==============================================

This module covers tokenization - the process of converting text to numbers
that LLMs can process.

Two phases:
1. TRAINING PHASE: Build the vocabulary (text → vocabulary)
2. INFERENCE PHASE: Use the vocabulary (text ↔ tokens)

Requirements:
    pip install tiktoken

Key concepts:
- Word-level vs Character-level vs Subword-level tokenization
- BPE (Byte Pair Encoding) algorithm
- Using tiktoken library
"""

import re
from collections import Counter, defaultdict


# =============================================================================
# PART 1: Text Splitting Approaches
# =============================================================================

def demonstrate_splitting_approaches():
    """Compare word-level, character-level, and subword-level tokenization."""
    
    print("="*70)
    print("TEXT SPLITTING APPROACHES")
    print("="*70)
    
    sample_text = "The transformer architecture revolutionized NLP. ChatGPT uses transformers!"
    
    print(f"\nSample text: \"{sample_text}\"\n")
    print("-"*70)
    
    # 1. Word-level tokenization
    print("\n1. WORD-LEVEL TOKENIZATION")
    print("   Method: Split on whitespace and punctuation")
    
    word_tokens = re.findall(r'\w+|[^\w\s]', sample_text)
    print(f"   Tokens: {word_tokens}")
    print(f"   Count: {len(word_tokens)} tokens")
    print("""
   ✓ Pros:
     - Intuitive, words are meaningful units
     - Smaller sequence lengths
   
   ✗ Cons:
     - HUGE vocabulary needed (100K+ words in English)
     - Can't handle unknown words ("OOV" problem)
     - Can't handle typos: "transformrs" → unknown
     - Different languages have different word boundaries
     - "running", "runs", "ran" are all separate tokens
    """)
    
    # 2. Character-level tokenization
    print("\n2. CHARACTER-LEVEL TOKENIZATION")
    print("   Method: Each character is a token")
    
    char_tokens = list(sample_text)
    print(f"   Tokens: {char_tokens[:30]}... (showing first 30)")
    print(f"   Count: {len(char_tokens)} tokens")
    print("""
   ✓ Pros:
     - Tiny vocabulary (~256 for ASCII, ~65K for Unicode)
     - Can handle ANY text, typos, new words
     - No OOV problem
   
   ✗ Cons:
     - VERY long sequences (each word = many tokens)
     - Characters don't carry much meaning alone
     - Hard to learn word-level patterns
     - Expensive: attention is O(n²) in sequence length
    """)
    
    # 3. Subword-level tokenization (the winner)
    print("\n3. SUBWORD-LEVEL TOKENIZATION (Modern Standard)")
    print("   Method: Learn common subword units from data (BPE, WordPiece, etc.)")
    
    # Simulated subword tokenization (actual BPE shown later)
    subword_tokens = ["The", " transform", "er", " architecture", " revolution", "ized", 
                      " NLP", ".", " Chat", "GPT", " uses", " transform", "ers", "!"]
    print(f"   Tokens: {subword_tokens}")
    print(f"   Count: {len(subword_tokens)} tokens")
    print("""
   ✓ Pros:
     - Balanced vocabulary size (~32K-100K)
     - Handles unknown words by breaking into known subwords
     - "transformers" → ["transform", "ers"]
     - "ChatGPT" → ["Chat", "GPT"] 
     - Typos handled: "transformrs" → ["transform", "rs"]
     - Efficient sequence lengths
   
   ✗ Cons:
     - More complex to implement
     - Vocabulary must be trained on representative data
     - Token boundaries can be non-intuitive
    """)
    
    print("\n" + "="*70)
    print("CONCLUSION: Subword tokenization (BPE) is the modern standard")
    print("="*70)
    print("""
    Why subword won:
    ┌─────────────┬────────────┬─────────────┬──────────────┐
    │ Approach    │ Vocab Size │ Seq Length  │ OOV Problem  │
    ├─────────────┼────────────┼─────────────┼──────────────┤
    │ Word        │ 100K+      │ Short       │ Severe       │
    │ Character   │ ~256       │ Very Long   │ None         │
    │ Subword     │ 32K-100K   │ Medium      │ Minimal      │
    └─────────────┴────────────┴─────────────┴──────────────┘
    
    Subword = best of both worlds!
    """)


# =============================================================================
# PART 2: BPE Algorithm Explanation
# =============================================================================

def demonstrate_bpe_algorithm():
    """Step-by-step demonstration of the BPE algorithm."""
    
    print("\n" + "="*70)
    print("BPE (BYTE PAIR ENCODING) ALGORITHM")
    print("="*70)
    print("""
    BPE is how we BUILD the subword vocabulary during training.
    
    Core idea: Start with characters, repeatedly merge the most frequent pair.
    """)
    
    # Training corpus
    corpus = ["low", "lower", "newest", "widest"]
    
    print(f"\nTraining corpus: {corpus}")
    print("\n" + "-"*70)
    print("STEP 1: Initialize with character vocabulary + end-of-word marker")
    print("-"*70)
    
    # Initialize: split each word into characters + end marker
    def get_vocab(corpus):
        vocab = defaultdict(int)
        for word in corpus:
            # Add space between chars, add </w> at end
            word_with_spaces = ' '.join(list(word)) + ' </w>'
            vocab[word_with_spaces] += 1
        return vocab
    
    vocab = get_vocab(corpus)
    print("\nInitial vocabulary (word frequencies):")
    for word, count in vocab.items():
        print(f"  '{word}': {count}")
    
    print("\nInitial tokens: ", set(' '.join(vocab.keys()).split()))
    
    # BPE iterations
    print("\n" + "-"*70)
    print("STEP 2: Iteratively merge most frequent pair")
    print("-"*70)
    
    def get_pair_counts(vocab):
        """Count frequency of adjacent pairs."""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def merge_pair(vocab, pair):
        """Merge all occurrences of pair in vocabulary."""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        return new_vocab
    
    # Run BPE for a few iterations
    num_merges = 10
    merges = []
    
    for i in range(num_merges):
        pairs = get_pair_counts(vocab)
        if not pairs:
            break
        
        # Find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        best_count = pairs[best_pair]
        
        print(f"\nIteration {i+1}:")
        print(f"  Most frequent pair: {best_pair} (count: {best_count})")
        
        # Merge
        vocab = merge_pair(vocab, best_pair)
        merges.append(best_pair)
        
        print(f"  New merge rule: {best_pair[0]} + {best_pair[1]} → {''.join(best_pair)}")
        print(f"  Vocabulary now:")
        for word, count in vocab.items():
            print(f"    '{word}': {count}")
    
    print("\n" + "-"*70)
    print("FINAL RESULT")
    print("-"*70)
    print(f"\nMerge rules learned (in order):")
    for i, (a, b) in enumerate(merges, 1):
        print(f"  {i}. {a} + {b} → {a}{b}")
    
    print(f"\nFinal vocabulary:")
    all_tokens = set()
    for word in vocab.keys():
        all_tokens.update(word.split())
    print(f"  {sorted(all_tokens)}")
    
    print("""
    
    Key insight: BPE learns common patterns from the training data:
    - 'e' and 's' merge early (common in English)
    - 'est</w>' merges (common suffix pattern)
    - 'low' stays together (common word/subword)
    
    During INFERENCE, we apply these rules to tokenize new text.
    """)


# =============================================================================
# PART 3: Using tiktoken (OpenAI's tokenizer)
# =============================================================================

def demonstrate_tiktoken():
    """Demonstrate the tiktoken library."""
    
    print("\n" + "="*70)
    print("USING TIKTOKEN (OpenAI's Tokenizer Library)")
    print("="*70)
    
    try:
        import tiktoken
        # Get encoder for GPT-4 / GPT-3.5-turbo
        print("\nLoading cl100k_base encoder (used by GPT-4, GPT-3.5-turbo)...")
        enc = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        print("""
    tiktoken not installed. Install with:
        pip install tiktoken
    
    Showing simulated output instead...
        """)
        demonstrate_tiktoken_simulated()
        return
    except Exception as e:
        print(f"""
    Could not load tiktoken encoder (network issue): {type(e).__name__}
    
    Showing simulated output instead...
        """)
        demonstrate_tiktoken_simulated()
        return
    
    # Basic info
    print(f"\n1. VOCABULARY SIZE")
    print(f"   enc.n_vocab = {enc.n_vocab:,} tokens")
    print("   (This is the total number of possible tokens)")
    
    # Encoding
    print(f"\n2. ENCODING (text → tokens)")
    print("-"*50)
    
    examples = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "ChatGPT",
        "antidisestablishmentarianism",
        "こんにちは",  # Japanese
        "🚀🎉",  # Emojis
        "def hello():\n    print('Hello')",  # Code
    ]
    
    for text in examples:
        tokens = enc.encode(text)
        print(f"\n   Text: \"{text}\"")
        print(f"   Token IDs: {tokens}")
        print(f"   Token count: {len(tokens)}")
        
        # Show individual tokens
        token_strings = [enc.decode([t]) for t in tokens]
        print(f"   Tokens: {token_strings}")
    
    # Decoding
    print(f"\n3. DECODING (tokens → text)")
    print("-"*50)
    
    token_ids = [9906, 11, 1917, 0]  # "Hello, world!"
    decoded = enc.decode(token_ids)
    print(f"\n   Token IDs: {token_ids}")
    print(f"   Decoded: \"{decoded}\"")
    
    # Round trip
    print(f"\n4. ROUND TRIP (encode → decode)")
    print("-"*50)
    
    original = "The transformer architecture is amazing!"
    encoded = enc.encode(original)
    decoded = enc.decode(encoded)
    
    print(f"\n   Original: \"{original}\"")
    print(f"   Encoded:  {encoded}")
    print(f"   Decoded:  \"{decoded}\"")
    print(f"   Match: {original == decoded}")
    
    # Different encoders
    print(f"\n5. DIFFERENT ENCODERS")
    print("-"*50)
    
    encoders = {
        "cl100k_base": "GPT-4, GPT-3.5-turbo, text-embedding-ada-002",
        "p50k_base": "Codex, code-davinci-002",
        "r50k_base": "GPT-3 (davinci, curie, babbage, ada)",
    }
    
    test_text = "Hello, how are you doing today?"
    
    print(f"\n   Text: \"{test_text}\"")
    print()
    for enc_name, models in encoders.items():
        try:
            encoder = tiktoken.get_encoding(enc_name)
            tokens = encoder.encode(test_text)
            print(f"   {enc_name}:")
            print(f"     Models: {models}")
            print(f"     Vocab size: {encoder.n_vocab:,}")
            print(f"     Token count: {len(tokens)}")
            print(f"     Tokens: {tokens}")
            print()
        except Exception as e:
            print(f"   {enc_name}: Could not load ({e})")
    
    # Practical insights
    print(f"\n6. PRACTICAL INSIGHTS")
    print("-"*50)
    
    print("""
    Token counting matters for:
    - API costs (charged per token)
    - Context window limits (e.g., 128K tokens for GPT-4)
    - Response length limits
    
    Rule of thumb:
    - ~1 token ≈ 4 characters in English
    - ~1 token ≈ ¾ of a word
    - 100 tokens ≈ 75 words
    """)
    
    # Token counting example
    text = "This is a sample paragraph to demonstrate token counting. " * 10
    tokens = enc.encode(text)
    chars = len(text)
    words = len(text.split())
    
    print(f"   Example text: {words} words, {chars} characters, {len(tokens)} tokens")
    print(f"   Ratio: {chars/len(tokens):.1f} chars/token, {words/len(tokens):.2f} words/token")


def demonstrate_tiktoken_simulated():
    """Simulated tiktoken output when library isn't available."""
    
    print("""
    SIMULATED TIKTOKEN OUTPUT
    ========================
    
    1. VOCABULARY SIZE
       enc.n_vocab = 100,277 tokens
       (This is the total number of possible tokens in cl100k_base)
    
    2. ENCODING (text → tokens)
    --------------------------------------------------
    
       Text: "Hello, world!"
       Token IDs: [9906, 11, 1917, 0]
       Token count: 4
       Tokens: ['Hello', ',', ' world', '!']
    
       Text: "ChatGPT"
       Token IDs: [34, 2143, 38]
       Token count: 3
       Tokens: ['Chat', 'G', 'PT']
    
       Text: "antidisestablishmentarianism"
       Token IDs: [519, 85, 2480, 4090, 309, 479, 8078, 2191]
       Token count: 8
       Tokens: ['ant', 'id', 'ises', 'tab', 'lish', 'ment', 'arian', 'ism']
    
       Text: "def hello():\n    print('Hello')"
       Token IDs: [755, 24748, 4019, 512, 262, 1426, 493, 15496, 873]
       Token count: 9
       Tokens: ['def', ' hello', '():', '\\n', '    ', 'print', "('", 'Hello', "')"]
    
    3. DECODING (tokens → text)
    --------------------------------------------------
    
       Token IDs: [9906, 11, 1917, 0]
       Decoded: "Hello, world!"
    
    4. KEY INSIGHT: The tokenizer breaks unknown/long words into subwords:
    
       "transformer" → ['trans', 'former'] (2 tokens)
       "transformers" → ['transform', 'ers'] (2 tokens)  
       "ChatGPT" → ['Chat', 'G', 'PT'] (3 tokens)
       
    This is BPE in action during inference!
    """)


# =============================================================================
# PART 4: Tokenization Connection to LLM Architecture
# =============================================================================

def explain_tokenization_in_llm_pipeline():
    """Explain how tokenization connects to the LLM pipeline."""
    
    print("\n" + "="*70)
    print("TOKENIZATION IN THE LLM PIPELINE")
    print("="*70)
    
    print("""
    Remember our earlier insight: "LLMs have never seen a single word"
    
    Here's the complete flow:
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     USER INPUT                                       │
    │                   "Hello, world!"                                    │
    └───────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     TOKENIZER                                        │
    │              "Hello, world!" → [9906, 11, 1917, 0]                   │
    │                                                                      │
    │   This is where text becomes numbers!                               │
    │   The LLM never sees "Hello" - only token ID 9906                   │
    └───────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                   EMBEDDING LAYER                                    │
    │         [9906, 11, 1917, 0] → [[0.02, -0.8, ...],                   │
    │                                [0.15, 0.33, ...],                   │
    │                                [0.77, -0.21, ...],                  │
    │                                [-0.5, 0.91, ...]]                   │
    │                                                                      │
    │   Each token ID → dense vector (e.g., 4096 dimensions)              │
    │   This is a learned lookup table                                    │
    └───────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                   TRANSFORMER LAYERS                                 │
    │                                                                      │
    │   Self-attention, feed-forward networks, layer norm...              │
    │   Processes the embeddings, captures relationships                  │
    │   This is where "understanding" emerges                            │
    └───────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                   OUTPUT LAYER                                       │
    │                                                                      │
    │   For each position, predict probability of EACH token              │
    │   Output: [vocab_size] probabilities = [100,277] numbers            │
    │   Highest probability → next token ID                               │
    └───────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                   DETOKENIZER                                        │
    │              [1722] → "Hi"                                           │
    │                                                                      │
    │   Simple lookup: token ID → string                                  │
    │   This is just a table lookup, not a neural network                │
    └─────────────────────────────────────────────────────────────────────┘
    
    
    KEY POINTS:
    
    1. Tokenization happens BEFORE the model sees anything
       - Same tokenizer for training and inference
       - Vocabulary is fixed after training
    
    2. The LLM's "vocabulary" = the tokenizer's vocabulary
       - cl100k_base has 100,277 tokens
       - Model's output layer predicts over these 100,277 options
    
    3. Token boundaries affect model behavior
       - "ChatGPT" → ['Chat', 'G', 'PT'] (3 tokens, not 1 word)
       - Model "sees" this as 3 separate units
       - This is why LLMs can struggle with character-level tasks
         (e.g., "How many r's in strawberry?")
    
    4. Different models use different tokenizers
       - GPT-4: cl100k_base (100K vocab)
       - LLaMA: SentencePiece (32K vocab)
       - Must use matching tokenizer for each model!
    """)


# =============================================================================
# PART 5: Hands-on Exercises
# =============================================================================

def print_exercises():
    """Print exercises for students."""
    
    print("\n" + "="*70)
    print("HANDS-ON EXERCISES")
    print("="*70)
    
    print("""
    Exercise 1: Token Counting
    --------------------------
    Using tiktoken, count the tokens in these texts:
    
    a) "The quick brown fox"
    b) "Supercalifragilisticexpialidocious"
    c) "Python is great! 🐍"
    d) Your own 100-word paragraph
    
    Calculate: chars/token ratio for each
    
    
    Exercise 2: Compare Tokenizers
    ------------------------------
    Compare how different encoders tokenize the same text:
    
    text = "def calculate_fibonacci(n):"
    
    Try with: cl100k_base, p50k_base, r50k_base
    Which produces fewer tokens? Why?
    
    
    Exercise 3: Multilingual Tokenization
    -------------------------------------
    Compare tokenization of the same meaning in different languages:
    
    English: "Hello, how are you?"
    Spanish: "Hola, ¿cómo estás?"
    Japanese: "こんにちは、お元気ですか？"
    
    Why do some languages produce more tokens?
    (Hint: What was the training data for BPE?)
    
    
    Exercise 4: Edge Cases
    ----------------------
    Try tokenizing these and explain the results:
    
    a) "aaaaaaaaaaaaaaaaaaa" (repeated character)
    b) "a a a a a a a a a a" (spaced characters)
    c) "123456789" (numbers)
    d) "   " (whitespace)
    e) "" (empty string)
    
    
    Exercise 5: Implement Basic BPE
    -------------------------------
    Given this corpus: ["low", "lower", "newest", "widest", "low", "low"]
    
    a) Initialize character vocabulary
    b) Run 5 BPE merge iterations by hand
    c) What merge rules did you learn?
    d) How would you tokenize "lowest" with your vocabulary?
    """)


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║     Tokenization for LLMs: Teaching Demonstration                    ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run all demonstrations
    demonstrate_splitting_approaches()
    demonstrate_bpe_algorithm()
    demonstrate_tiktoken()
    explain_tokenization_in_llm_pipeline()
    print_exercises()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
    Tokenization has TWO phases:
    
    1. TRAINING PHASE (build vocabulary):
       - Run BPE on training corpus
       - Learn merge rules from data
       - Output: vocabulary + merge rules
    
    2. INFERENCE PHASE (use vocabulary):
       - Apply merge rules to new text
       - text → token IDs (encode)
       - token IDs → text (decode)
    
    Key tools:
    - tiktoken: OpenAI's fast tokenizer library
    - tiktokenizer.vercel.app: Visual tokenization explorer
    - HuggingFace tokenizers: For research/custom tokenizers
    
    Remember: The LLM never sees words, only token IDs!
    """)
