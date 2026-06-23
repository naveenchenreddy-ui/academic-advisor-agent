"""
Ingest external URLs into Pinecone.

Add your college's website, career blogs, job boards, etc. to the
URLS_TO_INGEST dict below and run this script once to populate
the knowledge base with live web content.

Usage:
    python scripts/ingest_urls.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.academic_advisor_agent import AcademicAdvisorAgent

URLS_TO_INGEST = {
    "academic_guidance": [
        # "https://your-college.edu/academics/curriculum",
        # "https://your-college.edu/departments/cse/courses",
    ],
    "career_planning": [
        # "https://internshala.com/blog/career-guidance/",
        # "https://www.ambitionbox.com/career-advice",
    ],
    "job_market": [
        # "https://www.naukri.com/blog/job-market-trends/",
    ],
    "higher_studies": [
        # "https://www.shiksha.com/study-abroad",
        # "https://gate.iit.ac.in/",
    ],
}


if __name__ == "__main__":
    print("Initialising agent …")
    agent = AcademicAdvisorAgent()

    total = 0
    for category, urls in URLS_TO_INGEST.items():
        if urls:
            print(f"\nIngesting category: {category}")
            n = agent.ingest_urls(urls, category=category)
            total += n
        else:
            print(f"  (no URLs configured for '{category}' — skipping)")

    print(f"\n✅ Done. Total chunks ingested: {total}")
