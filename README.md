# 🎓 Academic Advisor Agent v2

AI-powered student advisor built with **LangChain + Pinecone + Claude 3.5 Sonnet**.
Gives meaningful, context-aware answers on courses, careers, job market, higher studies, grades, wellness, and admin.

---

## Folder structure

```
academic_advisor_full/
│
├── run.py                          ← Entry point (start here)
├── requirements.txt
├── .env.example                    ← Copy to .env and add keys
├── .gitignore
│
├── backend/
│   ├── __init__.py
│   ├── academic_advisor_agent.py   ← Main agent (7 roles, Claude-first)
│   ├── student_profile.py          ← Auto-extracts student context from chat
│   ├── embeddings.py               ← Hash embeddings (swap for OpenAI in prod)
│   └── scraper.py                  ← Web scraper for ingesting URLs
│
├── config/
│   ├── __init__.py
│   └── settings.py                 ← All config from .env
│
├── knowledge_base/
│   ├── __init__.py
│   ├── loader.py                   ← Loads all .txt docs into Pinecone
│   └── docs/
│       ├── academic_guidance.txt   ← Curriculum, credits, elective advice
│       ├── career_planning.txt     ← Job paths, internships, companies
│       ├── job_market.txt          ← 2024-2025 market trends, salaries
│       ├── higher_studies.txt      ← GATE, GRE, MS abroad, PhD, MBA
│       ├── wellness.txt            ← Stress, motivation, mental health
│       └── administrative.txt      ← Forms, deadlines, scholarships
│
├── scripts/
│   ├── ingest_urls.py              ← Scrape & ingest external URLs
│   └── clear_index.py              ← Wipe Pinecone index
│
└── tests/
    ├── __init__.py
    └── test_role_detection.py      ← Unit tests for role routing
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env:
#   ANTHROPIC_API_KEY=sk-ant-...
#   PINECONE_API_KEY=pcsk_...

# 3. Run the advisor
python run.py
```

---

## 7 advisor roles

| Role | Emoji | What it handles |
|---|---|---|
| Academic Guidance | 🎓 | Courses, electives, curriculum, semester planning |
| Career Planning | 🧭 | Job paths, resume, internships, skill development |
| Job Market | 📈 | 2025 hiring trends, salaries, hot skills, companies |
| Higher Studies | 🎯 | GATE, GRE, MS abroad, PhD, MBA, IIT MTech |
| Academic Progress | 📊 | CGPA, study strategies, exam preparation |
| Mentorship | 🧑‍🏫 | Stress, burnout, motivation, personal support |
| Administrative | 📝 | Forms, fees, scholarships, certificates |

---

## Example questions it answers well

```
"I'm in CSE semester 5, CGPA 7.4 — which electives for AI/ML?"
"Is data science still in demand in 2025 or should I do DevOps?"
"How do I prepare for GATE CS in 6 months from scratch?"
"Best universities for MS in ML — USA vs Germany comparison?"
"My attendance dropped to 68% — what do I do?"
"I feel completely burned out and want to give up"
"How do I apply for merit scholarship and what is the deadline?"
```

---

## Extending the knowledge base

**Add a new document:**
```
knowledge_base/docs/your_topic.txt
```
The filename becomes the category. Re-run `python run.py` and it auto-ingests.

**Scrape live websites:**
```bash
# Edit scripts/ingest_urls.py to add your URLs, then:
python scripts/ingest_urls.py
```

**Upgrade embeddings (recommended for production):**
```python
# In backend/embeddings.py, replace SimpleHashEmbeddings with:
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
```

---

## Architecture

```
Student query
     │
     ▼
Role Detector (weighted keyword scoring + profile context)
     │                          │
     ▼                          ▼
StudentProfile          Pinecone Vector DB
(auto-extracted)        (RAG: top-5 chunks)
     │                          │
     └──────────┬───────────────┘
                ▼
     Deep role-specific system prompt
     (expert knowledge + student context + RAG + history)
                │
                ▼
       Claude 3.5 Sonnet (max 2048 tokens)
                │
                ▼
     Structured, meaningful answer
```

---

## CLI commands

| Command | Action |
|---|---|
| Any question | Auto-routed to best role |
| `role:<id> <question>` | Force a specific role |
| `report` | Generate personalized progress report |
| `profile` | View auto-extracted student profile |
| `stats` | Show Pinecone index stats |
| `help` | Show command menu |
| `quit` | Exit |
