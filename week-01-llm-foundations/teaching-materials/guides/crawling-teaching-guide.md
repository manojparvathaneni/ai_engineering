# Web Crawling for LLM Training Data: Teaching Guide

## Overview

This module covers web crawling - the first step in the LLM data pipeline. Before a model can learn anything, you need data. Lots of it.

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                             │
│                                                                  │
│   Crawling  ──►  Data Cleaning  ──►  Tokenization               │
│   (this module)                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Model Architecture & Training
```

---

## Two Approaches to Crawling

| Aspect | Custom Crawlers (Big Labs) | Common Crawl (Public) |
|--------|---------------------------|----------------------|
| **Who uses it** | OpenAI, Google, Anthropic | Researchers, startups, open-source projects |
| **Control** | Full control over what/how/when | Take what's available |
| **Cost** | Expensive infrastructure | Free to download |
| **Freshness** | Can crawl continuously | Monthly snapshots |
| **Competitive edge** | Yes - proprietary data | No - everyone has access |
| **Cited in** | GPT papers (custom crawler) | LLaMA, Falcon, T5 papers |

### Custom Crawlers

Big labs build their own crawlers for several reasons:
- **Control**: Decide exactly what to crawl and how often
- **Quality**: Custom filtering at crawl time
- **Freshness**: Continuous updates, not monthly snapshots
- **Competitive advantage**: Data others don't have

### Common Crawl

A non-profit that crawls the web and makes data freely available:
- **Scale**: ~3.5 billion web pages per monthly crawl (~250TB compressed)
- **Format**: WARC files (raw HTML + metadata), WET files (extracted text)
- **History**: Archives going back to 2008
- **Usage**: Foundation for C4, The Pile, RefinedWeb, RedPajama datasets

---

## How a Web Crawler Works

### The Basic Algorithm

```
┌──────────────┐
│  Seed URLs   │  ← Starting points
└──────┬───────┘
       ▼
┌──────────────┐
│ URL Frontier │  ← Queue of URLs to visit
└──────┬───────┘
       ▼
┌──────────────┐     ┌──────────────┐
│  Fetch Page  │────►│   Visited?   │ ← Skip if already crawled
└──────┬───────┘     └──────────────┘
       ▼
┌──────────────┐
│ Extract Text │  ← Parse HTML, get content
└──────┬───────┘
       ▼
┌──────────────┐
│ Find Links   │  ← Discover new URLs
└──────┬───────┘
       │
       ├────► Store content
       │
       └────► Add new URLs to frontier
              │
              ▼
         (Repeat)
```

### Core Components

| Component | Purpose | Simple Implementation | Production Scale |
|-----------|---------|----------------------|------------------|
| **URL Frontier** | Queue of URLs to crawl | Python `deque` | Redis, Kafka |
| **Visited Set** | Track already-crawled URLs | Python `set` | Bloom filters, distributed DB |
| **Fetcher** | Download web pages | `requests` library | Thousands of parallel workers |
| **Parser** | Extract text from HTML | BeautifulSoup | Custom parsers per content type |
| **Storage** | Save crawled content | JSON files | S3, HDFS, data lakes |

---

## Distributed Crawler Architecture (Production Scale)

```
                                    ┌─────────────┐
                                    │  Seed URLs  │
                                    └──────┬──────┘
                                           ▼
                                    ┌─────────────┐
                                    │ URL Frontier│ ← Distributed queue
                                    │(Redis/Kafka)│   Billions of URLs
                                    └──────┬──────┘
                                           ▼
                 ┌─────────────────────────┼─────────────────────────┐
                 ▼                         ▼                         ▼
          ┌────────────┐            ┌────────────┐            ┌────────────┐
          │  Worker 1  │            │  Worker 2  │            │  Worker N  │
          │  (Fetch)   │            │  (Fetch)   │            │  (Fetch)   │
          └─────┬──────┘            └─────┬──────┘            └─────┬──────┘
                │                         │                         │
                │         Thousands of workers across many IPs      │
                └─────────────────────────┼─────────────────────────┘
                                          ▼
                                   ┌─────────────┐
                                   │   Parser    │ 
                                   └──────┬──────┘
                                          ▼
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
                   ┌─────────────┐                 ┌─────────────┐
                   │ Raw Storage │                 │ New URLs →  │
                   │  (S3/HDFS)  │                 │   Frontier  │
                   │  Petabytes  │                 └─────────────┘
                   └─────────────┘
```

### Scaling Challenges

| Challenge | Problem | Solution |
|-----------|---------|----------|
| **Politeness** | Can't hammer servers | Rate limiting, respect `robots.txt`, crawl-delay |
| **Distribution** | Single machine too slow | Thousands of parallel workers |
| **Deduplication** | Same content, different URLs | URL normalization, content hashing |
| **Prioritization** | Not all pages equal value | PageRank-style scoring, domain quality |
| **URL Traps** | Infinite URLs (calendars, search) | Depth limits, pattern detection |
| **Storage** | Petabytes of data | Compression, deduplication, tiered storage |

---

## Case Study: Building a Domain-Specific LLM

Let's make this concrete with an example: building an LLM specialized for **Microsoft Fabric**.

### Step 1: Identify Data Sources

| Source | URL Pattern | Content Type | Value |
|--------|-------------|--------------|-------|
| **Official Docs** | learn.microsoft.com/fabric/* | Technical reference | High - authoritative |
| **Blog Posts** | blog.fabric.microsoft.com/* | Tutorials, announcements | High - current |
| **Community Forums** | community.fabric.microsoft.com/* | Q&A, troubleshooting | High - real problems |
| **GitHub** | github.com/microsoft/fabric-samples | Code examples | Medium - practical |
| **Support Cases** | (internal) | Real issues + solutions | Very high - gold data |
| **Internal Wikis** | (internal) | Tribal knowledge | Very high - unique |
| **YouTube Transcripts** | youtube.com (Fabric channels) | Video tutorials | Medium - supplementary |

### Step 2: Configure the Crawler

```python
SEED_URLS = [
    "https://learn.microsoft.com/en-us/fabric/",
    "https://community.fabric.microsoft.com/",
    "https://blog.fabric.microsoft.com/",
]

ALLOWED_DOMAINS = [
    "learn.microsoft.com",
    "community.fabric.microsoft.com", 
    "blog.fabric.microsoft.com",
]

# URL filtering
INCLUDE_PATTERNS = ["/fabric/", "/Fabric/"]
EXCLUDE_PATTERNS = ["/locale/", "/pdf/", "/archive/", "/print/"]
```

### Step 3: Content-Specific Extraction

Different sources need different parsing strategies:

| Source | Extract | Ignore | Special Handling |
|--------|---------|--------|------------------|
| **Docs** | Article body, code blocks, headers | Nav, footer, sidebar, "Was this helpful?" | Preserve code formatting |
| **Forums** | Questions, accepted answers, vote counts | Ads, user profiles, pagination | Keep Q→A structure |
| **Blogs** | Post content, date, author | Related posts, share buttons | Track publish date |
| **GitHub** | README, code + comments, issues | CI configs, binary files | Preserve file structure |

### Step 4: What You End Up With

```
Microsoft Fabric Dataset
├── docs/           (~10K pages)    - Structured reference material
├── tutorials/      (~500 posts)    - Step-by-step guides  
├── qa_pairs/       (~50K pairs)    - Community Q&A (gold for fine-tuning!)
├── code_samples/   (~1K repos)     - Working examples
└── support_cases/  (~10K cases)    - Real-world problems + solutions

Total: ~100K documents, ~50M tokens
```

### Why Domain-Specific Crawling Matters

| General Web Crawl | Domain-Specific Crawl |
|-------------------|----------------------|
| Billions of pages | Thousands of pages |
| Mostly noise for your use case | Every page is relevant |
| Needs heavy filtering | Already filtered by source |
| Generic knowledge | Deep expertise |
| Expensive (compute + storage) | Cheap (focused scope) |

**Key insight**: For a Fabric assistant, 100K high-quality Fabric documents beats 1B random web pages.

---

## Ethical & Legal Considerations

### Technical Compliance

| Mechanism | What It Is | Your Responsibility |
|-----------|------------|---------------------|
| **robots.txt** | File telling crawlers what not to access | Respect it (legally gray if you don't) |
| **Crawl-delay** | Requested time between requests | Honor it to avoid overloading servers |
| **Terms of Service** | Website rules | Varies - some prohibit crawling |

### Ongoing Legal Debates

- **Copyright**: Can you train on copyrighted content? (NYT vs OpenAI, Getty vs Stability AI)
- **Opt-out**: Should sites be able to block AI training crawlers?
- **Data licensing**: Some content requires attribution or payment
- **PII**: Personal information scattered in web content

### Best Practices for Your Students

1. **Be a good citizen**: Respect robots.txt, use reasonable delays
2. **Document your sources**: Know where your training data came from
3. **Consider licensing**: Some content is explicitly licensed for reuse
4. **Public vs. Private**: Public web ≠ permission to use commercially

---

## Connection to Next Steps

Crawling produces **raw text**. But it's messy:
- Duplicate content everywhere
- Boilerplate (nav, footers, ads)
- Low-quality pages (spam, SEO garbage)
- Mixed languages
- PII scattered throughout

This leads directly to the next topic: **Data Cleaning**.

```
Crawling          Data Cleaning         Tokenization
(raw HTML)  ───►  (clean text)    ───►  (tokens)
   │                   │                    │
   │                   │                    │
   ▼                   ▼                    ▼
 Messy             Filtered            Training-ready
```

---

## Discussion Questions for Students

1. **Why might OpenAI build their own crawler instead of just using Common Crawl?**
   
2. **What biases might exist in crawled data?** (Think: language, geography, topics, perspectives)

3. **How does the choice of seed URLs affect what the model learns?**

4. **For a domain-specific model, when is it better to:**
   - Fine-tune an existing model on your crawled data?
   - Train from scratch on your data?

5. **What are the ethical implications of training on:**
   - Public forum posts?
   - Copyrighted books?
   - Social media content?

---

## Hands-On Exercise

> **Design a crawl strategy for a domain-specific LLM**
> 
> Pick a domain (a product, field, or company) and:
> 1. Identify 5+ data sources you'd crawl
> 2. Specify URL patterns to include/exclude
> 3. Describe what content to extract vs. ignore for each source
> 4. Estimate the size of your resulting dataset
> 5. Identify potential data quality issues

---

## Key Takeaways

1. **Crawling is the first step** - no data, no model
2. **Two approaches**: Custom crawlers (control, cost) vs. Common Crawl (free, shared)
3. **Scale matters**: Frontier models use trillions of tokens from billions of pages
4. **Quality > Quantity for domain models**: Focused crawling beats broad crawling
5. **Crawling is just the start**: Raw data needs cleaning and tokenization before training

---

## Code Reference

See: `simple_crawler_demo.py` - A working Python crawler demonstrating:
- URL frontier (queue management)
- Domain restriction
- Content extraction
- Link discovery
- Politeness (rate limiting)

---

*Next module: Data Cleaning - Turning raw crawled data into training-ready text*
