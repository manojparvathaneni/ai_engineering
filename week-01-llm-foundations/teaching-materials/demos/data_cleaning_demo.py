"""
Exploring LLM Training Datasets - Demonstration
================================================

This script demonstrates the key concepts of data cleaning for LLM training:
1. Quality filters (Gopher rules, C4 rules)
2. Deduplication (n-gram Jaccard similarity)
3. Sample data from major datasets

For actually loading datasets from HuggingFace, see the full version
with: pip install datasets huggingface_hub
"""


# =============================================================================
# Quality Filter Demonstrations
# =============================================================================

def demonstrate_quality_filters():
    """Show how quality filters work on sample text."""
    
    print("\n" + "="*70)
    print("QUALITY FILTER DEMONSTRATIONS")
    print("="*70)
    print("""
    These are the types of heuristic filters used by C4, Gopher, and FineWeb
    to separate high-quality text from junk.
    """)
    
    # Sample texts: good vs bad
    samples = {
        "good_article": """
        Machine learning has transformed how we approach complex problems in science 
        and engineering. By learning patterns from data, these algorithms can make 
        predictions and decisions that were previously impossible to automate.
        
        The field has grown rapidly since the introduction of deep learning, with 
        applications ranging from computer vision to natural language processing.
        Recent advances in large language models have opened new possibilities for
        human-computer interaction and automated reasoning.
        """,
        
        "spam_content": """
        CLICK HERE NOW!!! Best deals on AMAZING products!!!
        $$$ MAKE MONEY FAST $$$ 
        Visit www.totally-not-spam.com for FREE stuff!!!
        BUY NOW BUY NOW BUY NOW
        Limited time offer!!!! Act NOW!!!!
        """,
        
        "boilerplate": """
        Home | About | Contact | Privacy Policy | Terms of Service
        Click here to subscribe to our newsletter
        Share on Facebook | Share on Twitter | Share on LinkedIn
        Copyright 2024 All Rights Reserved
        Cookie Policy | Sitemap | Accessibility
        """,
        
        "garbled_text": """
        asdkfjh askdjfh 823742 kdsjfh sdf987 @#$%^&* 
        lorem ipsum dolor sit amet consectetur
        ........ ........ ........ 
        AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        xyzzy plugh 12345 67890
        """,
        
        "short_incomplete": """
        Error 404
        Page not found
        """,
        
        "forum_post": """
        Question: How do I fix the "connection refused" error in Python requests?
        
        I keep getting this error when trying to connect to my local server.
        Any help would be appreciated!
        
        Answer: This usually means the server isn't running or is listening on 
        a different port. Try checking if your server is actually started, and
        verify the port number matches what you're connecting to.
        """,
    }
    
    print("Applying quality filters to sample texts:\n")
    print("-" * 70)
    
    for name, text in samples.items():
        text = text.strip()
        words = text.split()
        chars = len(text)
        lines = text.count('\n') + 1
        
        # Calculate statistics
        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
        alpha_chars = sum(c.isalpha() for c in text)
        alpha_ratio = alpha_chars / chars if chars else 0
        upper_chars = sum(c.isupper() for c in text)
        upper_ratio = upper_chars / chars if chars else 0
        
        # Count lines ending with ellipsis or exclamation
        lines_list = text.split('\n')
        ellipsis_lines = sum(1 for l in lines_list if l.strip().endswith('...'))
        exclaim_lines = sum(1 for l in lines_list if l.strip().endswith('!'))
        
        # Check for HTML
        has_html = '<' in text and '>' in text
        
        # Quality checks (based on Gopher/C4/FineWeb rules)
        checks = {
            'word_count (50-100k)': 50 <= len(words) <= 100000,
            'avg_word_len (3-10)': 3 <= avg_word_len <= 10,
            'alpha_ratio (>70%)': alpha_ratio > 0.7,
            'upper_ratio (<30%)': upper_ratio < 0.3,
            'no_html_tags': not has_html,
        }
        
        passed = sum(checks.values())
        total = len(checks)
        verdict = "✓ KEEP" if passed >= 4 else "✗ FILTER OUT"
        
        print(f"\n{name.upper()}")
        print(f"  Preview: {text[:60].replace(chr(10), ' ')}...")
        print(f"  Stats: {len(words)} words | avg_len={avg_word_len:.1f} | alpha={alpha_ratio:.0%} | upper={upper_ratio:.0%}")
        print(f"  Checks ({passed}/{total}):")
        for check_name, check_passed in checks.items():
            status = "✓" if check_passed else "✗"
            print(f"    {status} {check_name}")
        print(f"  → {verdict}")


# =============================================================================
# Deduplication Demonstration
# =============================================================================

def demonstrate_deduplication():
    """Show how deduplication works with n-gram Jaccard similarity."""
    
    print("\n\n" + "="*70)
    print("DEDUPLICATION DEMONSTRATION")
    print("="*70)
    print("""
    Deduplication removes repeated content that would cause the model to
    memorize specific text. This uses MinHash/Jaccard similarity.
    """)
    
    # Example documents with varying similarity
    documents = [
        ("Original article", 
         "The quick brown fox jumps over the lazy dog. This is a classic pangram used for testing."),
        ("Near duplicate (minor edit)", 
         "The quick brown fox jumps over the lazy dog. This is a famous pangram used for testing."),
        ("Semantic duplicate (reworded)", 
         "A fast auburn fox leaps above a sleepy canine. This classic sentence tests typing."),
        ("Different topic", 
         "Machine learning is a subset of artificial intelligence that enables computers to learn."),
        ("Exact duplicate", 
         "The quick brown fox jumps over the lazy dog. This is a classic pangram used for testing."),
    ]
    
    def get_ngrams(text, n=3):
        """Get word n-grams from text."""
        words = text.lower().split()
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
    
    def jaccard_similarity(set1, set2):
        """Calculate Jaccard similarity: |intersection| / |union|"""
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0
    
    print("\nDocuments:")
    print("-" * 70)
    for i, (label, doc) in enumerate(documents):
        print(f"  [{i}] {label}")
        print(f"      \"{doc[:60]}...\"")
    
    print("\n\nPairwise Jaccard Similarity (using 3-grams):")
    print("-" * 70)
    
    ngrams = [get_ngrams(doc, n=3) for _, doc in documents]
    
    # Show the n-grams for first document
    print(f"\n  Example 3-grams from doc [0]:")
    sample_ngrams = list(ngrams[0])[:5]
    for ng in sample_ngrams:
        print(f"    \"{ng}\"")
    print(f"    ... ({len(ngrams[0])} total)")
    
    print("\n  Similarity scores:")
    duplicates = []
    for i in range(len(documents)):
        for j in range(i+1, len(documents)):
            sim = jaccard_similarity(ngrams[i], ngrams[j])
            if sim > 0.8:
                status = "🔴 EXACT/NEAR DUPLICATE"
                duplicates.append(j)
            elif sim > 0.5:
                status = "🟡 Similar"
            else:
                status = "🟢 Different"
            print(f"    [{i}] vs [{j}]: {sim:6.1%}  {status}")
    
    print(f"\n  Deduplication result (threshold=0.8):")
    print(f"    Keep: [0], [2], [3]")
    print(f"    Remove: [1] (near-dup of [0]), [4] (exact dup of [0])")


# =============================================================================
# Sample Data from Major Datasets
# =============================================================================

def show_sample_data():
    """Show what actual data from these datasets looks like."""
    
    print("\n\n" + "="*70)
    print("SAMPLE DATA FROM MAJOR DATASETS")
    print("="*70)
    
    # These are representative examples of what you'd find in each dataset
    
    samples = {
        "C4": {
            "text": """Photosynthesis is the process by which plants convert light energy into chemical energy. This process occurs in the chloroplasts of plant cells, where chlorophyll absorbs sunlight. The light energy is used to convert carbon dioxide and water into glucose and oxygen. This fundamental biological process is essential for life on Earth, as it produces both food for plants and oxygen for animals to breathe.""",
            "metadata": {"url": "example.edu/biology/photosynthesis", "timestamp": "2019-04"},
            "notes": "Clean, educational content. C4 used aggressive filtering."
        },
        
        "RefinedWeb": {
            "text": """Getting Started with Docker Containers

Docker has revolutionized how developers build, ship, and run applications. In this tutorial, we'll walk through the basics of containerization.

First, install Docker on your system. For Ubuntu:

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

Once installed, verify with:

docker --version

Containers provide isolation, portability, and consistency across development and production environments.""",
            "metadata": {"url": "techblog.example.com/docker-intro", "dump_id": "CC-2023-14"},
            "notes": "RefinedWeb preserves code blocks and technical content well."
        },
        
        "FineWeb": {
            "text": """The history of jazz music spans over a century, originating in the African American communities of New Orleans in the late 19th and early 20th centuries. Jazz emerged from a blend of African and European musical traditions, incorporating blues, ragtime, and brass band music.

Key figures like Louis Armstrong, Duke Ellington, and Charlie Parker shaped the genre through innovation and virtuosity. The music evolved through various styles including swing, bebop, cool jazz, and fusion.

Today, jazz continues to influence contemporary music and remains a vital art form celebrated worldwide.""",
            "metadata": {"url": "music-history.example.org/jazz", "score": 0.89, "language": "en"},
            "notes": "FineWeb includes quality scores from their filtering pipeline."
        },
        
        "Dolma (Wikipedia subset)": {
            "text": """Alan Turing (23 June 1912 – 7 June 1954) was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist. Turing was highly influential in the development of theoretical computer science, providing a formalization of the concepts of algorithm and computation with the Turing machine.

During the Second World War, Turing worked for the Government Code and Cypher School at Bletchley Park, Britain's codebreaking centre. He devised techniques for speeding the breaking of German ciphers.""",
            "metadata": {"source": "wikipedia", "subset": "en"},
            "notes": "Dolma includes Wikipedia as a high-quality subset."
        },
        
        "Dolma (Reddit subset)": {
            "text": """ELI5: Why does ice float on water when most solids sink?

When water freezes, the molecules form a crystal structure that takes up more space than liquid water. This makes ice less dense than liquid water (about 9% less dense).

Since ice is less dense, it floats! This is actually unusual - most substances are denser as solids. This property is crucial for life on Earth because it means lakes freeze from the top down, allowing fish to survive underneath.""",
            "metadata": {"source": "reddit", "subreddit": "explainlikeimfive", "score": 847},
            "notes": "Dolma includes Reddit as conversational/educational content."
        },
        
        "The Pile (arXiv subset)": {
            "text": """Abstract: We present a novel approach to neural machine translation using attention mechanisms. Our model, which we call the Transformer, relies entirely on self-attention to compute representations of its input and output. Experiments on English-to-German and English-to-French translation tasks show that these models achieve state-of-the-art results while being more parallelizable and requiring significantly less time to train.

1. Introduction

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...""",
            "metadata": {"source": "arxiv", "category": "cs.CL"},
            "notes": "The Pile included academic papers for technical knowledge."
        },
    }
    
    for dataset_name, data in samples.items():
        print(f"\n{'─'*70}")
        print(f"📁 {dataset_name}")
        print(f"{'─'*70}")
        print(f"Metadata: {data['metadata']}")
        print(f"Note: {data['notes']}")
        print(f"\nText sample:")
        print(f"  {data['text'][:400]}...")
        print(f"\n  [{len(data['text'])} chars, {len(data['text'].split())} words]")


# =============================================================================
# Dataset Comparison Summary
# =============================================================================

def print_dataset_summary():
    """Print a summary comparison of all major datasets."""
    
    print("\n\n" + "="*70)
    print("DATASET COMPARISON SUMMARY")
    print("="*70)
    
    print("""
    ┌─────────────┬────────────┬─────────────────┬───────────────────────────┐
    │ Dataset     │ Size       │ Source          │ Key Innovation            │
    ├─────────────┼────────────┼─────────────────┼───────────────────────────┤
    │ C4          │ ~156B tok  │ Common Crawl    │ First major cleaned web   │
    │ The Pile    │ ~300B tok  │ 22 sources      │ Diverse curation          │
    │ RefinedWeb  │ ~5T tok    │ Common Crawl    │ Web-only can be excellent │
    │ Dolma       │ ~3T tok    │ Multi-source    │ Full transparency         │
    │ FineWeb     │ ~15T tok   │ Common Crawl    │ Comprehensive ablations   │
    └─────────────┴────────────┴─────────────────┴───────────────────────────┘
    """)
    
    print("""
    Key Papers to Read:
    ───────────────────
    1. C4/T5:      arxiv.org/abs/1910.10683  (Section 2: "The Colossal Clean Crawled Corpus")
    2. The Pile:   arxiv.org/abs/2101.00027  ("An 800GB Dataset of Diverse Text")
    3. RefinedWeb: arxiv.org/abs/2306.01116  ("The RefinedWeb Dataset for Falcon LLM")
    4. Dolma:      arxiv.org/abs/2402.00159  ("An Open Corpus of Three Trillion Tokens")
    5. FineWeb:    huggingface.co/datasets/HuggingFaceFW/fineweb (Blog + Dataset Card)
    """)
    
    print("""
    Evolution of Cleaning Approaches:
    ─────────────────────────────────
    
    C4 (2019)      Simple heuristics, aggressive filtering
        ↓              └─ Removed too much good content
        ↓          
    The Pile       Curated multi-source approach
    (2020)             └─ Quality through source selection
        ↓          
    RefinedWeb     Advanced web filtering
    (2023)             └─ Proved web-only can match curated
        ↓          
    FineWeb        Systematic ablations + best practices
    (2024)             └─ Current state-of-the-art pipeline
    """)


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║     LLM Training Data Cleaning: Concepts & Demonstrations            ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run all demonstrations
    demonstrate_quality_filters()
    demonstrate_deduplication()
    show_sample_data()
    print_dataset_summary()
    
    print("\n" + "="*70)
    print("TO EXPLORE ACTUAL DATASETS:")
    print("="*70)
    print("""
    Install the HuggingFace datasets library:
    
        pip install datasets huggingface_hub
    
    Then load datasets in streaming mode (no full download):
    
        from datasets import load_dataset
        
        # FineWeb (recommended starting point)
        ds = load_dataset("HuggingFaceFW/fineweb-edu", 
                          name="sample-10BT", 
                          split="train", 
                          streaming=True)
        
        for sample in ds:
            print(sample['text'][:500])
            break
    
    Streaming mode lets you explore without downloading terabytes!
    """)
