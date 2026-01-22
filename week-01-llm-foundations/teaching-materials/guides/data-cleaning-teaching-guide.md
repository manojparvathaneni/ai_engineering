# Data Cleaning for LLM Training: Teaching Guide

## Overview

Data cleaning is where the real quality of your training data is determined. Crawling gives you raw material; cleaning determines what the model actually learns.

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                             │
│                                                                  │
│   Crawling  ──►  Data Cleaning  ──►  Tokenization               │
│                  (this module)                                   │
└─────────────────────────────────────────────────────────────────┘
```

**The core insight**: Garbage in, garbage out. Quality cleaning can make web data competitive with curated sources.

---

## Why Data Cleaning Matters

### The Duplicate Problem

| Scenario | Impact |
|----------|--------|
| Same news story on 1000 sites | Model memorizes this specific text |
| Boilerplate ("Subscribe to newsletter") | Appears in every response |
| Template pages | Model learns filler, not content |
| Copy-paste content | Skewed probability distribution |

**Real example**: If "Click here to subscribe" appears 10M times in training data, the model learns this is important text to generate.

### The Quality Problem

| Bad Data | Consequence |
|----------|-------------|
| Spam/SEO content | Poor writing, repetitive patterns |
| Machine-generated text | "AI slop" gets amplified |
| OCR errors | Misspellings, gibberish |
| Non-text (code errors, logs) | Incoherent outputs |

### The Safety Problem

| Concern | Risk |
|---------|------|
| Toxic content | Model learns harmful language |
| PII (names, emails, SSNs) | Model can leak personal info |
| Copyrighted material | Legal liability |
| Malware/exploits | Security risks |

---

## Major Training Datasets

### Evolution of Approaches

```
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
```

### Dataset Comparison

| Dataset | Size | Source | Innovation | Paper |
|---------|------|--------|------------|-------|
| **C4** | ~156B tokens | Common Crawl | First major cleaned corpus | arxiv.org/abs/1910.10683 |
| **The Pile** | ~300B tokens | 22 sources | Diversity through curation | arxiv.org/abs/2101.00027 |
| **RefinedWeb** | ~5T tokens | Common Crawl | Web-only excellence | arxiv.org/abs/2306.01116 |
| **Dolma** | ~3T tokens | Multi-source | Full transparency | arxiv.org/abs/2402.00159 |
| **FineWeb** | ~15T tokens | Common Crawl | Ablation studies | HuggingFace blog |

---

## Deep Dive: C4 (Colossal Clean Crawled Corpus)

**Source**: Google's T5 paper (2019)
**Used by**: T5, early LLMs

### Cleaning Pipeline

```
Common Crawl (April 2019)
        ↓
    Language filter (English only via langdetect)
        ↓
    Remove pages with "bad words" (blocklist)
        ↓
    Remove pages with "lorem ipsum"
        ↓
    Keep only lines ending in punctuation
        ↓
    Deduplicate (3-sentence level)
        ↓
    C4 Dataset (~156B tokens)
```

### C4's Problems (discovered later)

| Issue | Example |
|-------|---------|
| Too aggressive | Removed legitimate health discussions (blocklist had "sex") |
| Punctuation rule | Removed code, lists, dialogue |
| Simple heuristics | Missed sophisticated spam |
| English-only filter | Noisy for non-English |

**Lesson**: Simple rules have unintended consequences at scale.

---

## Deep Dive: The Pile

**Source**: EleutherAI (2020)
**Used by**: GPT-Neo, GPT-J, GPT-NeoX

### Philosophy: Diversity Through Curation

Instead of cleaning one source well, combine many high-quality sources:

| Component | Size | Content |
|-----------|------|---------|
| Pile-CC | 227GB | Common Crawl subset |
| PubMed Central | 90GB | Medical papers |
| Books3 | 101GB | Books (controversial) |
| OpenWebText2 | 62GB | Reddit-linked content |
| ArXiv | 56GB | Scientific papers |
| GitHub | 95GB | Code |
| Wikipedia | 17GB | Encyclopedia |
| StackExchange | 32GB | Q&A |
| USPTO | 23GB | Patents |
| FreeLaw | 51GB | Legal opinions |
| DM Mathematics | 8GB | Math problems |
| Ubuntu IRC | 6GB | Technical chat |
| ... | ... | 10 more sources |

### Key Insight

Different sources bring different qualities:
- Wikipedia → Factual, well-structured
- ArXiv → Technical/scientific reasoning
- GitHub → Code understanding
- StackExchange → Q&A format
- Books → Long-form coherence

**Lesson**: Source diversity can matter more than source size.

---

## Deep Dive: RefinedWeb

**Source**: Technology Innovation Institute (Falcon team, 2023)
**Used by**: Falcon models

### Key Contribution

**Proved that web-only data can match curated multi-source data** if cleaned well enough.

### The RefinedWeb Pipeline

```
Common Crawl (multiple dumps)
        │
        ▼
┌───────────────────────┐
│   URL Filtering       │ ← Blocklists (adult, spam domains)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│   Text Extraction     │ ← trafilatura (better than jusText)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Language Detection   │ ← fastText classifier
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│   Quality Filters     │ ← Heuristics + perplexity
│   • Gopher rules      │
│   • C4 rules (fixed)  │
│   • Repetition filter │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│    Deduplication      │
│   • MinHash (fuzzy)   │ ← Document-level
│   • Line-level dedup  │ ← Removes repeated boilerplate
└───────────┬───────────┘
            ▼
    RefinedWeb (~5T tokens)
```

### Key Finding from Paper

> "Web data filtered with our pipeline can match or exceed the performance of curated datasets on downstream tasks."

This was surprising - people assumed you needed books, Wikipedia, etc.

---

## Deep Dive: Dolma

**Source**: AI2 / Allen Institute (2024)
**Used by**: OLMo models

### Key Contribution

**Full transparency** - released the entire pipeline, not just data.

### Multi-Source Composition

| Source | Tokens | Notes |
|--------|--------|-------|
| Common Crawl | 2.3T | Web (heavily cleaned) |
| The Stack | 411B | Code |
| Reddit | 89B | Conversational |
| Semantic Scholar | 57B | Academic |
| Wikipedia + Wikibooks | 4.3B | Encyclopedia |
| Project Gutenberg | 5B | Books |

### The Dolma Toolkit

AI2 released `dolma` as a Python package:

```python
# Example: Use Dolma's taggers
from dolma import taggers

# Tag documents with quality scores
tagger = taggers.GopherQualityTagger()
results = tagger.tag(documents)
```

**Lesson**: Transparency enables community improvement.

---

## Deep Dive: FineWeb (Current State-of-the-Art)

**Source**: HuggingFace (2024)
**Size**: ~15T tokens from 96 Common Crawl dumps

### Why FineWeb Matters

1. **Most comprehensive cleaning study** - ablations showing impact of each step
2. **Released everything** - data, code, methodology
3. **Becoming the new standard** - most recent open models use it

### The FineWeb Pipeline

```
96 Common Crawl Dumps (2013-2024)
              │
              ▼
     ┌────────────────────┐
     │   URL Filtering    │
     │  • Adult blocklist │
     │  • Spam domains    │
     └─────────┬──────────┘
               ▼
     ┌────────────────────┐
     │  Text Extraction   │
     │  (trafilatura)     │
     └─────────┬──────────┘
               ▼
     ┌────────────────────┐
     │ Language Detection │
     │    (fastText)      │
     └─────────┬──────────┘
               ▼
     ┌─────────────────────────────────────────┐
     │         Quality Filtering               │
     │                                         │
     │  Line-level:                            │
     │   • Remove lines that are all caps      │
     │   • Remove lines with no alphabetic     │
     │   • Remove short lines (<10 chars)      │
     │                                         │
     │  Document-level:                        │
     │   • Word count (50 - 100,000)           │
     │   • Mean word length (3-10 chars)       │
     │   • Symbol-to-word ratio (<0.1)         │
     │   • Alphabetic character ratio (>0.8)   │
     │   • Stop word ratio (>0.06)             │
     │                                         │
     │  Repetition removal:                    │
     │   • Top 2-gram ratio (<0.2)             │
     │   • Top 3-gram ratio (<0.18)            │
     │   • Top 4-gram ratio (<0.16)            │
     │   • Duplicate line ratio (<0.3)         │
     │   • Duplicate paragraph ratio (<0.3)    │
     │                                         │
     │  C4-style filters (adapted)             │
     │  Gopher-style filters                   │
     └─────────────────┬───────────────────────┘
                       ▼
     ┌────────────────────┐
     │   Deduplication    │
     │  • MinHash (docs)  │
     │  • 5-gram shingles │
     │  • Jaccard > 0.8   │
     └─────────┬──────────┘
               ▼
       FineWeb (~15T tokens)
```

### FineWeb Ablation Results

HuggingFace tested the impact of each filter:

| Filter | Impact on Downstream Tasks |
|--------|---------------------------|
| Deduplication | ↑↑↑ Large improvement |
| Quality filters (combined) | ↑↑ Significant improvement |
| Language filtering | ↑ Moderate improvement |
| URL blocklists | ↑ Modest improvement |

**Key finding**: Deduplication has the largest single impact.

### FineWeb-Edu

A subset of FineWeb filtered for educational content:
- Used a classifier to score "educational value"
- Trained on annotations from Llama-3-70B
- ~1.3T high-quality educational tokens

---

## Cleaning Strategies In-Depth

### 1. Deduplication

**Why it's critical**: Largest single impact on model quality.

#### Approaches

| Method | How It Works | Scale | Catches |
|--------|--------------|-------|---------|
| **Exact hash** | SHA256 of document | Billions | Exact copies |
| **MinHash/LSH** | Approximate similarity | Billions | Near-duplicates |
| **N-gram overlap** | Compare n-gram sets | Millions | Partial overlap |
| **Suffix arrays** | Find repeated substrings | Millions | Small repetitions |
| **Embedding similarity** | Semantic matching | Thousands | Paraphrases |

#### MinHash Explained

```python
# Simplified MinHash concept

def create_shingles(text, n=5):
    """Create n-gram shingles from text"""
    words = text.lower().split()
    return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

def minhash_signature(shingles, num_hashes=128):
    """Create MinHash signature"""
    signature = []
    for seed in range(num_hashes):
        min_hash = min(hash(shingle + str(seed)) for shingle in shingles)
        signature.append(min_hash)
    return signature

def estimate_similarity(sig1, sig2):
    """Estimate Jaccard similarity from signatures"""
    matches = sum(s1 == s2 for s1, s2 in zip(sig1, sig2))
    return matches / len(sig1)

# Two documents are near-duplicates if similarity > 0.8
```

#### FineWeb's Deduplication Settings

- **Shingle size**: 5-grams (5 consecutive words)
- **Signature size**: 128 hashes
- **Threshold**: Jaccard similarity > 0.8
- **Scope**: Within each Common Crawl dump + across dumps

---

### 2. Quality Filtering

#### Heuristic Rules (Fast, Scalable)

**"Gopher" Rules** (from DeepMind):

```python
def passes_gopher_rules(text):
    words = text.split()
    
    # Word count bounds
    if not (50 <= len(words) <= 100000):
        return False
    
    # Mean word length
    mean_word_len = sum(len(w) for w in words) / len(words)
    if not (3 <= mean_word_len <= 10):
        return False
    
    # Fraction of words with alphabetic char
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_words / len(words) < 0.8:
        return False
    
    # Lines ending with ellipsis
    lines = text.split('\n')
    ellipsis_lines = sum(1 for l in lines if l.strip().endswith('...'))
    if ellipsis_lines / len(lines) > 0.3:
        return False
    
    # Lines starting with bullet
    bullet_lines = sum(1 for l in lines if l.strip().startswith('•'))
    if bullet_lines / len(lines) > 0.9:
        return False
    
    return True
```

**FineWeb Repetition Filters**:

```python
def check_repetition(text):
    words = text.lower().split()
    
    # Count n-gram frequencies
    def get_top_ngram_ratio(words, n):
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        if not ngrams:
            return 0
        from collections import Counter
        counts = Counter(ngrams)
        most_common_count = counts.most_common(1)[0][1]
        return most_common_count / len(ngrams)
    
    # Thresholds from FineWeb
    if get_top_ngram_ratio(words, 2) > 0.20:  # Top 2-gram too frequent
        return False
    if get_top_ngram_ratio(words, 3) > 0.18:  # Top 3-gram too frequent
        return False
    if get_top_ngram_ratio(words, 4) > 0.16:  # Top 4-gram too frequent
        return False
    
    return True
```

#### Model-Based Filtering (Slower, More Accurate)

| Method | How It Works | When to Use |
|--------|--------------|-------------|
| **Perplexity scoring** | Score with small LM, remove high-perplexity | Final cleanup pass |
| **Quality classifier** | Train classifier on good/bad examples | When you have labeled data |
| **LLM-as-judge** | Use large model to score content | For FineWeb-Edu style filtering |

**FineWeb-Edu Classifier**:
1. Sample documents from FineWeb
2. Have Llama-3-70B rate "educational value" (0-5)
3. Train small classifier on these ratings
4. Apply to all of FineWeb
5. Keep documents scoring > 3

---

### 3. Content Safety Filtering

| Category | Method | Action |
|----------|--------|--------|
| **Adult content** | URL blocklists + keyword detection | Remove |
| **Toxic language** | Classifier (Perspective API, custom) | Remove/flag |
| **PII** | Regex + NER models | Mask or remove |
| **Malware snippets** | Pattern matching | Remove |

**PII Detection Patterns**:

```python
import re

PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
}

def mask_pii(text):
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f'[{pii_type.upper()}]', text)
    return text
```

---

### 4. Text Extraction

Not all extractors are equal:

| Tool | Quality | Speed | Best For |
|------|---------|-------|----------|
| BeautifulSoup + get_text() | Low | Fast | Quick and dirty |
| jusText | Medium | Medium | C4-era standard |
| **trafilatura** | High | Medium | Current best practice |
| readability-lxml | Medium | Fast | News articles |

**trafilatura** is now the standard because it:
- Removes boilerplate (nav, ads, footers)
- Preserves article structure
- Handles diverse layouts
- Extracts metadata (title, date, author)

```python
import trafilatura

html = fetch_page(url)
text = trafilatura.extract(html, include_comments=False, include_tables=True)
```

---

## Hands-On: Exploring Datasets

### Loading Datasets (Streaming Mode)

```python
from datasets import load_dataset

# FineWeb (recommended starting point)
ds = load_dataset(
    "HuggingFaceFW/fineweb-edu", 
    name="sample-10BT", 
    split="train", 
    streaming=True  # Don't download everything
)

for i, sample in enumerate(ds):
    print(sample['text'][:500])
    if i >= 5:
        break
```

### Analyzing Sample Quality

```python
def analyze_document(text):
    words = text.split()
    lines = text.split('\n')
    
    stats = {
        'word_count': len(words),
        'char_count': len(text),
        'avg_word_length': sum(len(w) for w in words) / len(words),
        'line_count': len(lines),
        'alpha_ratio': sum(c.isalpha() for c in text) / len(text),
        'digit_ratio': sum(c.isdigit() for c in text) / len(text),
    }
    
    return stats
```

---

## Discussion Questions

1. **Why did RefinedWeb's finding (web-only can match curated) surprise people?**

2. **What are the tradeoffs between heuristic filters and model-based filters?**

3. **For a domain-specific model (e.g., Microsoft Fabric), would you:**
   - Apply the same filters as FineWeb?
   - Customize filters for technical content?
   - Skip filtering and trust your sources?

4. **How might deduplication affect model creativity vs. memorization?**

5. **What ethical considerations arise from:**
   - Removing "toxic" content (who decides what's toxic?)
   - Filtering for "educational" content (whose education?)
   - Masking PII (vs. removing documents with PII entirely)

---

## Key Takeaways

1. **Data quality > data quantity** - Well-cleaned smaller data beats poorly-cleaned larger data

2. **Deduplication is crucial** - Single largest impact on model quality

3. **Simple rules scale** - Heuristics filter billions of documents; models are for refinement

4. **Source diversity helps** - Different sources contribute different capabilities

5. **Transparency matters** - Dolma and FineWeb sharing pipelines enables community improvement

6. **It's evolving fast** - Best practices changed significantly from C4 (2019) to FineWeb (2024)

---

## Papers to Read

| Paper | Year | Key Contribution |
|-------|------|------------------|
| T5/C4 Paper | 2019 | First major cleaning study (Section 2) |
| The Pile | 2020 | Multi-source curation approach |
| Deduplicating Training Data | 2022 | Impact of deduplication |
| RefinedWeb | 2023 | Web-only can match curated |
| Dolma | 2024 | Transparent pipeline release |
| FineWeb | 2024 | Comprehensive ablations |

---

## Code Reference

See: `data_cleaning_demo.py` - Demonstrates:
- Quality filter rules (Gopher, C4, FineWeb-style)
- Deduplication with n-gram Jaccard similarity
- Sample data from each major dataset

---

*Next module: Tokenization - Converting cleaned text into training-ready tokens*
