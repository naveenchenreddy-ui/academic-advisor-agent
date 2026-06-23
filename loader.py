"""
"""Knowledge base loader — reads all .txt files from the current directory
and ingests them into ChromaDB via the agent's ingest_text() method.

Each file is named after its category (e.g. career_planning.txt)
so the category metadata is set automatically from the filename.
"""
from pathlib import Path


def load_all(agent) -> int:
    """
    Load every .txt file in the current directory into the vector store.
    Returns total number of chunks ingested.
    """
    docs_dir = Path(__file__).parent
    total = 0

    for fpath in sorted(docs_dir.glob("*.txt")):
        category = fpath.stem   # filename without extension = category name
        text = fpath.read_text(encoding="utf-8")
        label = category.replace("_", " ").title()
        n = agent.ingest_text(text, title=f"KB: {label}", category=category)
        total += n

    return total
