"""
Simple Web Crawler for LLM Data Collection
==========================================

This is an educational crawler demonstrating the core concepts
of web crawling for LLM training data collection.

Concepts demonstrated:
1. URL frontier (queue of URLs to visit)
2. Visited tracking (avoid re-crawling)
3. Domain restriction (stay within allowed domains)
4. Politeness (rate limiting with delays)
5. Content extraction (text from HTML)
6. Link discovery (find new URLs to crawl)

Usage:
    python simple_crawler.py

Requirements:
    pip install requests beautifulsoup4
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import json
from datetime import datetime


class SimpleCrawler:
    """
    A simple educational web crawler.
    
    In production crawlers (like those at OpenAI, Google, Anthropic):
    - URL frontier would be distributed (Redis, Kafka)
    - Multiple workers would fetch in parallel
    - Storage would be S3/HDFS, not local files
    - Much more sophisticated error handling
    """
    
    def __init__(self, seed_urls, allowed_domains, max_pages=10, delay=1.0):
        """
        Initialize the crawler.
        
        Args:
            seed_urls: Starting URLs to crawl
            allowed_domains: Only crawl URLs from these domains
            max_pages: Maximum number of pages to crawl
            delay: Seconds to wait between requests (politeness)
        """
        # URL Frontier: queue of URLs to visit
        self.frontier = deque(seed_urls)
        
        # Track visited URLs to avoid duplicates
        self.visited = set()
        
        # Domain restriction
        self.allowed_domains = allowed_domains
        
        # Politeness settings
        self.max_pages = max_pages
        self.delay = delay
        
        # Storage for crawled data
        self.crawled_data = []
        
        # Simple user agent (be honest about what you are)
        self.headers = {
            'User-Agent': 'EducationalCrawler/1.0 (Learning about LLM data collection)'
        }
    
    def is_allowed(self, url):
        """Check if URL is within allowed domains."""
        parsed = urlparse(url)
        return any(domain in parsed.netloc for domain in self.allowed_domains)
    
    def extract_text(self, soup):
        """
        Extract clean text from HTML.
        
        In production, you'd have much more sophisticated extraction:
        - Remove boilerplate (headers, footers, nav)
        - Preserve code blocks with formatting
        - Handle tables, lists specially
        - Extract metadata (title, date, author)
        """
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Basic cleaning: collapse multiple newlines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)
    
    def extract_links(self, soup, base_url):
        """
        Extract all links from the page.
        
        Returns only links within allowed domains.
        """
        links = []
        for anchor in soup.find_all('a', href=True):
            # Convert relative URLs to absolute
            url = urljoin(base_url, anchor['href'])
            
            # Clean URL (remove fragments)
            url = url.split('#')[0]
            
            # Only include allowed domains
            if self.is_allowed(url) and url not in self.visited:
                links.append(url)
        
        return list(set(links))  # Deduplicate
    
    def fetch_page(self, url):
        """
        Fetch a single page.
        
        Returns (html, status_code) or (None, error_message)
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text, response.status_code
        except requests.RequestException as e:
            return None, str(e)
    
    def crawl_page(self, url):
        """
        Crawl a single page: fetch, extract text, find links.
        """
        print(f"  Fetching: {url}")
        
        html, status = self.fetch_page(url)
        
        if html is None:
            print(f"  ❌ Error: {status}")
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "No title"
        
        # Extract text content
        text = self.extract_text(soup)
        
        # Extract links for further crawling
        new_links = self.extract_links(soup, url)
        
        # Store the crawled data
        self.crawled_data.append({
            'url': url,
            'title': title.strip() if title else '',
            'text': text,
            'text_length': len(text),
            'links_found': len(new_links),
            'crawled_at': datetime.now().isoformat()
        })
        
        print(f"  ✓ Title: {title[:50]}...")
        print(f"  ✓ Extracted {len(text)} characters, found {len(new_links)} links")
        
        return new_links
    
    def crawl(self):
        """
        Main crawl loop.
        
        This is the core algorithm:
        1. Take URL from frontier
        2. Fetch and process page
        3. Add new URLs to frontier
        4. Repeat until done
        """
        print("=" * 60)
        print("Starting crawler")
        print(f"Seed URLs: {list(self.frontier)}")
        print(f"Max pages: {self.max_pages}")
        print(f"Delay: {self.delay}s between requests")
        print("=" * 60)
        
        pages_crawled = 0
        
        while self.frontier and pages_crawled < self.max_pages:
            # Get next URL from frontier
            url = self.frontier.popleft()
            
            # Skip if already visited
            if url in self.visited:
                continue
            
            # Mark as visited
            self.visited.add(url)
            pages_crawled += 1
            
            print(f"\n[{pages_crawled}/{self.max_pages}] Crawling...")
            
            # Crawl the page
            new_links = self.crawl_page(url)
            
            # Add new links to frontier
            for link in new_links:
                if link not in self.visited:
                    self.frontier.append(link)
            
            # Politeness: wait before next request
            if self.frontier and pages_crawled < self.max_pages:
                print(f"  ⏳ Waiting {self.delay}s...")
                time.sleep(self.delay)
        
        print("\n" + "=" * 60)
        print(f"Crawling complete!")
        print(f"Pages crawled: {pages_crawled}")
        print(f"URLs remaining in frontier: {len(self.frontier)}")
        print("=" * 60)
        
        return self.crawled_data
    
    def save_results(self, filename):
        """Save crawled data to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.crawled_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(self.crawled_data)} pages to {filename}")
    
    def print_summary(self):
        """Print a summary of crawled content."""
        print("\n" + "=" * 60)
        print("CRAWL SUMMARY")
        print("=" * 60)
        
        if not self.crawled_data:
            print("No pages were crawled.")
            return
        
        total_chars = 0
        for i, page in enumerate(self.crawled_data, 1):
            print(f"\n{i}. {page['title'][:60]}")
            print(f"   URL: {page['url']}")
            print(f"   Text: {page['text_length']} chars")
            print(f"   Preview: {page['text'][:100]}...")
            total_chars += page['text_length']
        
        print("\n" + "-" * 60)
        print(f"Total pages: {len(self.crawled_data)}")
        print(f"Total text: {total_chars:,} characters")
        print(f"Average per page: {total_chars // len(self.crawled_data):,} characters")


# =============================================================================
# DEMO MODE: Simulated crawl with realistic sample data
# =============================================================================

def run_demo_mode():
    """
    Demonstrates the crawler output with simulated data.
    Use this when you can't access external URLs.
    """
    print("=" * 60)
    print("DEMO MODE: Simulating Microsoft Fabric documentation crawl")
    print("=" * 60)
    
    # Simulated crawl data (what a real crawl would produce)
    mock_crawled_data = [
        {
            "url": "https://learn.microsoft.com/en-us/fabric/get-started/",
            "title": "What is Microsoft Fabric? - Microsoft Fabric",
            "text": """What is Microsoft Fabric?
Microsoft Fabric is an all-in-one analytics solution for enterprises that covers 
everything from data movement to data science, Real-Time Analytics, and business 
intelligence. It offers a comprehensive suite of services, including data lake, 
data engineering, and data integration, all in one place.

Fabric integrates technologies like Azure Data Factory, Azure Synapse Analytics, 
and Power BI into a single unified product, empowering data and business professionals 
alike to unlock the potential of their data.

Key capabilities:
- Data Engineering: Build and manage data pipelines
- Data Warehouse: Store and query structured data at scale
- Data Science: Build and deploy machine learning models
- Real-Time Analytics: Analyze streaming data in real-time
- Power BI: Create interactive reports and dashboards""",
            "text_length": 892,
            "links_found": 15,
            "crawled_at": datetime.now().isoformat()
        },
        {
            "url": "https://learn.microsoft.com/en-us/fabric/data-engineering/",
            "title": "Data Engineering in Microsoft Fabric",
            "text": """Data Engineering in Microsoft Fabric
Data engineering in Microsoft Fabric enables users to design, build, and maintain 
infrastructures and systems that enable their organizations to collect, store, 
process, and analyze large volumes of data.

Lakehouse architecture
The lakehouse combines the best of data lakes and data warehouses. Store all your 
data in open Delta Lake format while getting warehouse-like query performance.

Spark compute
Use Apache Spark for large-scale data processing. Fabric provides managed Spark 
clusters that auto-scale based on your workload.

Data pipelines
Build ETL/ELT pipelines using a visual interface or code. Schedule and monitor 
your data workflows with built-in orchestration.""",
            "text_length": 743,
            "links_found": 12,
            "crawled_at": datetime.now().isoformat()
        },
        {
            "url": "https://learn.microsoft.com/en-us/fabric/data-warehouse/",
            "title": "Data Warehouse in Microsoft Fabric",
            "text": """Synapse Data Warehouse in Microsoft Fabric
The Synapse Data Warehouse in Microsoft Fabric provides a modern, cloud-native 
warehouse solution that enables you to store and analyze data at scale.

T-SQL Support
Write queries using familiar T-SQL syntax. The warehouse supports most T-SQL 
commands, making it easy for SQL developers to get started.

Automatic optimization
Fabric automatically optimizes your queries and manages storage. No need to 
manually tune indexes or manage file compaction.

Integration with Power BI
Connect directly to Power BI for reporting. Data stays in place - no need to 
export or move data for visualization.""",
            "text_length": 654,
            "links_found": 8,
            "crawled_at": datetime.now().isoformat()
        },
        {
            "url": "https://community.fabric.microsoft.com/t5/General/How-to-connect/td-p/123456",
            "title": "How to connect Fabric to on-premises data? - Community Forum",
            "text": """Question: How to connect Fabric to on-premises data?
Posted by: DataEngineer_Mike
Date: 2024-01-15

I'm trying to connect my Microsoft Fabric lakehouse to our on-premises SQL Server. 
What's the best approach?

---

Accepted Answer by: FabricExpert_Sarah

You have a few options:

1. On-premises data gateway
   - Install the gateway on a machine in your network
   - Configure it in Fabric admin portal
   - Create a connection using the gateway

2. Azure ExpressRoute
   - For high-volume, low-latency connections
   - Requires Azure networking setup

3. VNet data gateway (preview)
   - Managed gateway that runs in your Azure VNet
   - No on-premises installation needed

For most cases, I recommend starting with the on-premises data gateway. 
It's the simplest to set up and works well for moderate data volumes.

Helpful votes: 47""",
            "text_length": 921,
            "links_found": 5,
            "crawled_at": datetime.now().isoformat()
        },
        {
            "url": "https://blog.fabric.microsoft.com/en-us/blog/announcing-fabric-capacities/",
            "title": "Announcing new Fabric capacity options - Microsoft Fabric Blog",
            "text": """Announcing new Microsoft Fabric capacity options
Published: January 2024
Author: Fabric Product Team

We're excited to announce new capacity options for Microsoft Fabric that make 
it easier than ever to get started and scale your analytics workloads.

New F2 SKU
Starting at just $0.36/hour, the new F2 SKU is perfect for development, testing, 
and small production workloads. It includes:
- 2 capacity units
- Full access to all Fabric workloads
- Pay-as-you-go billing

Capacity autoscaling
Your Fabric capacity can now automatically scale up during peak usage and scale 
down during quiet periods. This ensures you always have the compute you need 
while optimizing costs.

Reservation discounts
Save up to 40% with 1-year reservations on Fabric capacity. Contact your 
Microsoft account team for details.""",
            "text_length": 834,
            "links_found": 7,
            "crawled_at": datetime.now().isoformat()
        }
    ]
    
    # Simulate the crawl process
    print("\nSimulating crawl with 1 second delays...\n")
    
    for i, page in enumerate(mock_crawled_data, 1):
        print(f"[{i}/{len(mock_crawled_data)}] Crawling...")
        print(f"  Fetching: {page['url']}")
        time.sleep(0.5)  # Shorter delay for demo
        print(f"  ✓ Title: {page['title'][:50]}...")
        print(f"  ✓ Extracted {page['text_length']} characters, found {page['links_found']} links")
        if i < len(mock_crawled_data):
            print(f"  ⏳ Waiting 1s...")
            time.sleep(0.5)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CRAWL SUMMARY")
    print("=" * 60)
    
    total_chars = 0
    for i, page in enumerate(mock_crawled_data, 1):
        print(f"\n{i}. {page['title'][:60]}")
        print(f"   URL: {page['url']}")
        print(f"   Text: {page['text_length']} chars")
        print(f"   Preview: {page['text'][:80]}...")
        total_chars += page['text_length']
    
    print("\n" + "-" * 60)
    print(f"Total pages: {len(mock_crawled_data)}")
    print(f"Total text: {total_chars:,} characters")
    print(f"Average per page: {total_chars // len(mock_crawled_data):,} characters")
    
    # Save results
    output_file = "fabric_crawl_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mock_crawled_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(mock_crawled_data)} pages to {output_file}")
    
    # Show what the data looks like
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT (what gets saved to JSON)")
    print("=" * 60)
    print(json.dumps(mock_crawled_data[0], indent=2)[:500] + "...")
    
    return mock_crawled_data


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     Simple Web Crawler for LLM Data Collection           ║
    ║     Educational Demonstration                            ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Run demo mode with simulated Fabric data
    results = run_demo_mode()
    
    print("\n" + "=" * 60)
    print("WHAT HAPPENS NEXT (in a real LLM pipeline)")
    print("=" * 60)
    print("""
    1. DATA CLEANING
       - Remove boilerplate, duplicates, low-quality pages
       - Filter by language, content type
       - Handle special content (code blocks, tables)
       
    2. TOKENIZATION  
       - Convert cleaned text to tokens
       - Build or apply vocabulary
       - Create training sequences
       
    3. TRAINING
       - Feed tokens to the model
       - Learn to predict next token
       - For domain models: fine-tune on this specialized data
       
    This crawler collected raw text. The cleaning and tokenization
    steps transform it into training-ready data.
    """)
    
    print("=" * 60)
    print("TO USE THIS CRAWLER FOR REAL:")
    print("=" * 60)
    print("""
    # 1. Install dependencies
    pip install requests beautifulsoup4
    
    # 2. Modify the configuration:
    SEED_URLS = ["https://learn.microsoft.com/en-us/fabric/"]
    ALLOWED_DOMAINS = ["learn.microsoft.com"]
    
    # 3. Create crawler and run
    crawler = SimpleCrawler(
        seed_urls=SEED_URLS,
        allowed_domains=ALLOWED_DOMAINS,
        max_pages=100,
        delay=2.0  # Be polite!
    )
    results = crawler.crawl()
    crawler.save_results("my_crawl.json")
    """)
