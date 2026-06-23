"""
Web scraper — fetches URLs and returns LangChain Documents.
Used by the agent to ingest external knowledge (college websites,
career blogs, job portals, etc.) into Pinecone.
"""
import time
from typing import Optional, List
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from langchain.schema import Document


def scrape_url(url: str, category: str = "general", timeout: int = 15) -> Optional[Document]:
    """Fetch a single URL and return a LangChain Document, or None on failure."""
    headers = {"User-Agent": "Mozilla/5.0 (AcademicAdvisor/2.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ⚠  Could not fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    content_node = soup.find("article") or soup.find("main") or soup.find("body")
    raw_text = content_node.get_text(separator="\n", strip=True) if content_node else ""
    title = soup.title.string.strip() if soup.title else url

    if len(raw_text) < 50:
        print(f"  ⚠  Too little content at {url}")
        return None

    print(f"  ✓  [{category}] '{title[:60]}' ({len(raw_text):,} chars)")
    return Document(
        page_content=raw_text,
        metadata={
            "source": url,
            "title": title,
            "category": category,
            "scraped_at": datetime.now().isoformat(),
        }
    )


def scrape_urls(urls: List[str], category: str = "general") -> List[Document]:
    """Scrape multiple URLs with a polite delay between requests."""
    docs = []
    for url in urls:
        doc = scrape_url(url, category=category)
        if doc:
            docs.append(doc)
        time.sleep(0.5)
    return docs
