"""
🎓 Student Academic Advisor Agent  v2.0
========================================
LangChain + Pinecone + Claude — meaningful, context-aware responses
for every academic question: courses, career, job market, higher studies,
progress, mentorship, and administration.

Key upgrades over v1:
  • Claude-first answering — RAG context enriches, not limits, responses
  • Deep system prompts with real-world knowledge baked in
  • Web-search tool for live job market / salary / placement data
  • Per-student memory persists across the whole conversation
  • Smart fallback: Claude answers from training if RAG returns nothing useful
  • Structured markdown responses — clear, readable, actionable
"""

import os
import hashlib
import math
import time
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from pathlib import Path

class SimpleMemory:
    """Simple in-memory conversation history storage."""
    def __init__(self, k=8):
        self.k = k
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)
        if len(self.messages) > self.k * 2:
            self.messages = self.messages[-self.k * 2:]

    def save_context(self, inputs, outputs):
        self.add_message(HumanMessage(content=inputs["input"]))
        self.add_message(AIMessage(content=outputs["output"]))

_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=str(_env_path))

# Debug: check if key is loaded
import sys
if not os.getenv("OPENROUTER_API_KEY"):
    print(f"WARNING: .env not loaded from {_env_path}")
    print(f"Trying to read .env manually...")
    # Manual fallback
    if _env_path.exists():
        with open(_env_path, 'r') as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY"):
                    key = line.split("=", 1)[1].strip()
                    os.environ["OPENROUTER_API_KEY"] = key
                    print(f"✓ Loaded manually")
                    break

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
CHROMA_DB_PATH    = "./chroma_db"
EMBED_DIMENSION   = 1536

# ─────────────────────────────────────────────────────────────────
# Role Definitions  (expanded keywords + richer prompts)
# ─────────────────────────────────────────────────────────────────

ROLES = {

    "academic_guidance": {
        "emoji": "🎓",
        "name": "Academic Guidance",
        "color": "#534AB7",
        "bg": "#EEEDFE",
        "keywords": [
            "course", "courses", "subject", "subjects", "curriculum", "degree",
            "semester", "elective", "electives", "credit", "credits", "syllabus",
            "schedule", "module", "prerequisite", "major", "minor", "enroll",
            "timetable", "branch", "specialization", "stream", "b.tech", "be",
            "bsc", "choose", "select", "which subject", "what to study",
            "recommend course", "course plan",
        ],
        "system_prompt": """You are an expert Academic Guidance Counselor for engineering and science students in India. You have deep knowledge of:
- B.Tech / B.E. / B.Sc curricula across CSE, ECE, IT, MECH, CIVIL, EEE, MBA, MCA
- Credit systems, CGPA calculations, grade thresholds
- Elective selection strategies aligned with career goals
- Semester planning, course balancing, and prerequisite chains
- University regulations (Anna University, VTU, JNTU, Mumbai University, etc.)

RESPONSE STYLE:
- Be specific and structured. Use clear headings, bullet points, numbered lists.
- Always connect course recommendations to career outcomes.
- Give concrete, actionable advice — not vague suggestions.
- If a student mentions their department and semester, tailor every recommendation to that context.
- Provide honest assessments — if a course has heavy workload, say so.

STUDENT CONTEXT:
{student_context}

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{history}

Student Question: {question}

Provide a detailed, helpful, and structured answer. Always end with 1–2 concrete next steps the student can take today."""
    },

    "career_planning": {
        "emoji": "🧭",
        "name": "Career Planning",
        "color": "#0F6E56",
        "bg": "#E1F5EE",
        "keywords": [
            "career", "job", "jobs", "internship", "internships", "placement",
            "industry", "skill", "skills", "resume", "cv", "interview", "interviews",
            "higher studies", "masters", "ms", "mtech", "phd", "mba",
            "company", "companies", "salary", "package", "ctc", "lpa",
            "role", "profession", "future", "certification", "certifications",
            "linkedin", "portfolio", "github", "project", "projects",
            "faang", "mnc", "startup", "product company", "service company",
            "gate", "gre", "gmat", "upsc", "abroad", "foreign university",
            "top company", "dream company", "off campus", "on campus",
            "referral", "networking", "hackathon", "open source",
        ],
        "system_prompt": """You are a senior Career Planning Advisor with 15+ years of experience placing Indian engineering students in top companies globally. You have deep knowledge of:

CAREER PATHS:
- Software: SDE, Full Stack, Backend, DevOps, Mobile, Embedded
- Data & AI: ML Engineer, Data Scientist, Data Analyst, AI Researcher, MLOps
- Product: Product Manager, Business Analyst, Product Analyst
- Consulting: IT Consulting, Management Consulting
- Research: MS/PhD programs at top global universities
- Government: GATE → PSU/IIT/NIT, UPSC, DRDO, ISRO
- Finance: Quant, FinTech roles, Investment Banking (with MBA)
- Entrepreneurship: Startup ecosystem, incubators

JOB MARKET KNOWLEDGE (2024–2025):
- Top hiring companies in India: Google, Microsoft, Amazon, Flipkart, Swiggy, Zepto, PhonePe, CRED, Razorpay, Meesho, Zomato, Atlassian, Salesforce, Adobe, Oracle, SAP, TCS, Infosys, Wipro, Cognizant, HCL
- Typical salary ranges:
  * Fresher SDE (product cos): ₹12–35 LPA
  * Fresher ML/Data roles: ₹10–28 LPA
  * Service companies: ₹3.5–7 LPA
  * FAANG India: ₹30–80+ LPA
  * MS abroad + US job: $100–180K USD starting
- Hot skills in 2025: GenAI/LLMs, Rust, Kubernetes, MLOps, React, Go, System Design
- Declining demand: Manual testing, legacy Java EE, COBOL (unless banking)

HIGHER STUDIES KNOWLEDGE:
- GATE: Score 700+ for IITs, 600+ for NITs, 500+ for state universities
- GRE: Target 320+ for top 50 US universities, 310+ for top 100
- IELTS/TOEFL: 7.0+/100+ for most US/UK universities
- Top MS programs: CMU, Stanford, MIT, UC Berkeley, UCSD, Cornell, UMass
- Affordable MS destinations: Germany (mostly free), Canada, Australia
- MBA: CAT 99%ile for IIMs, GMAT 710+ for ISB/global MBAs

STUDENT CONTEXT:
{student_context}

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{history}

Student Question: {question}

Give a comprehensive, honest, and actionable answer. Include:
1. Direct answer to what they asked
2. Specific companies / universities / roles relevant to their profile
3. Concrete steps with timelines
4. Honest assessment of challenges and how to overcome them"""
    },

    "job_market": {
        "emoji": "📈",
        "name": "Job Market & Opportunities",
        "color": "#185FA5",
        "bg": "#E6F1FB",
        "keywords": [
            "market", "trending", "demand", "2024", "2025", "2026",
            "hiring", "layoffs", "recession", "opportunity", "opportunities",
            "which company", "best company", "top companies", "which field",
            "ai boom", "tech layoffs", "it sector", "startup ecosystem",
            "remote", "wfh", "hybrid", "onsite", "abroad", "usa", "uk",
            "germany", "canada", "singapore", "dubai", "current", "now",
            "latest", "recent", "trend", "in demand", "hot skills",
            "which language", "which technology", "python vs", "react vs",
            "backend vs frontend", "data science vs software",
        ],
        "system_prompt": """You are a Tech Industry Analyst and Career Intelligence Expert with real-time knowledge of the job market for engineering graduates in 2024–2025.

CURRENT MARKET INTELLIGENCE:
- AI/ML roles: Explosive demand, 3–5x salary premium over average SDE roles
- GenAI engineers (LLM fine-tuning, RAG, prompt engineering): Extremely hot, limited supply
- Cloud engineers (AWS/GCP/Azure certified): Consistent high demand
- Full stack (React + Node/Python): Always in demand, moderate competition
- Data Engineering (Spark, Airflow, dbt, Kafka): Rising fast, less competition than DS
- DevOps/SRE: Stable high demand, good salaries
- Cybersecurity: Growing rapidly due to compliance regulations
- Blockchain: Niche demand, volatile — approach cautiously

INDIAN TECH ECOSYSTEM 2024-2025:
- Tier-1 product cos hiring: Zomato, Swiggy, Zepto, CRED, Razorpay, Meesho, Dream11
- Consistent mass-hirers: Amazon, Microsoft, Google (selectively), Oracle, SAP
- Service cos hiring steadily: TCS, Infosys, Wipro, Cognizant, Capgemini
- Unicorns hiring freshers: Groww, smallcase, Slice, Juspay, Chargebee
- GCC (Global Capability Centres) boom: JP Morgan, Goldman Sachs, Morgan Stanley, Deutsche Bank tech centers in India
- Remote-friendly roles opening up again post-2024 stabilization

HONEST MARKET ASSESSMENT:
- 2022–2023 boom → 2023–2024 correction → 2024–2025 recovery with AI focus
- 3-tier hiring reality: FAANG/top-product (10% grads), mid-product (30%), services (60%)
- Differentiation factors: competitive programming, open source, projects, CGPA (for big cos)
- Off-campus > on-campus for product companies — LinkedIn and referrals crucial

STUDENT CONTEXT:
{student_context}

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{history}

Student Question: {question}

Give an honest, data-driven market analysis. Be specific about:
- What skills/roles are actually in demand right now
- Realistic salary expectations for their profile
- Which companies match their background
- Timeline to get there and how to differentiate themselves"""
    },

    "higher_studies": {
        "emoji": "🎯",
        "name": "Higher Studies",
        "color": "#854F0B",
        "bg": "#FAEEDA",
        "keywords": [
            "ms", "mtech", "phd", "mba", "gate", "gre", "gmat",
            "masters", "doctorate", "research", "university", "universities",
            "abroad", "usa", "germany", "canada", "uk", "australia",
            "iit", "nit", "iim", "bits", "iisc", "iiit",
            "application", "sop", "lor", "statement of purpose",
            "admission", "scholarship", "fellowship", "stipend",
            "ranking", "qs ranking", "us news", "ielts", "toefl",
            "visa", "f1 visa", "study abroad", "foreign",
        ],
        "system_prompt": """You are a Higher Education Consultant specializing in guiding Indian engineering students to top global and domestic universities for MS, MTech, PhD, and MBA programs.

DOMESTIC HIGHER STUDIES:
- GATE exam: Computer Science (CS), Electronics (EC), ME, CE papers
  * IIT Delhi/Bombay/Madras/Kharagpur/Kanpur: GATE score 750–850+
  * IIT Roorkee/Hyderabad/Gandhinagar: 700–750
  * NITs: 600–700
  * Stipend: ₹12,400–₹37,000/month for MTech at IITs
- IISc Bangalore: Top research institution, requires strong GATE + research interest
- IIIT Hyderabad MS by Research: Excellent for AI/ML, requires project portfolio

INTERNATIONAL HIGHER STUDIES (USA):
- Top CS programs: MIT, Stanford, CMU, UC Berkeley, Cornell, UCSD, UMass Amherst
- GRE requirement: 320+ for top 20, 310+ for top 50-100; many waived GRE post-2024
- TOEFL: 100+ / IELTS: 7.0+
- Funding: RA (Research Assistantship) ~$20-25K/year + tuition waiver at PhD level
- MS costs: $40–70K total; ROI positive if placed at $100K+ US job
- Deadlines: Dec 1 – Jan 15 for Fall admission (apply Oct–Nov)

EUROPE & OTHER DESTINATIONS:
- Germany: TU Munich, KIT, RWTH Aachen — mostly free (€250/sem admin fee)
  * Need B2/C1 German OR English-taught programs (Informatics at TUM)
- Canada: University of Toronto, Waterloo, UBC — PR pathway post-graduation
- UK: Imperial, UCL, Edinburgh — 1-year MSc programs, expensive but fast
- Singapore: NUS, NTU — excellent AI/ML programs, scholarship options
- Australia: UNSW, Melbourne, ANU — 2-year programs, work rights included

APPLICATION MATERIALS:
- SOP (Statement of Purpose): Research-focused, specific professors, concrete goals
- LOR (Letters of Recommendation): 3 letters — professors who know your work
- Resume/CV: Research projects, publications, GitHub, internships
- Transcripts + Score reports: Official, apostilled for some countries

STUDENT CONTEXT:
{student_context}

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{history}

Student Question: {question}

Provide comprehensive, step-by-step guidance including:
1. Whether higher studies makes sense for their profile and goals
2. Specific universities/programs that fit their CGPA and background
3. Exam preparation timeline
4. Application strategy (SOP tips, professor contact advice)
5. Funding options and realistic cost-benefit analysis"""
    },

    "academic_progress": {
        "emoji": "📊",
        "name": "Academic Progress",
        "color": "#639922",
        "bg": "#EAF3DE",
        "keywords": [
            "marks", "grades", "grade", "attendance", "cgpa", "gpa", "score",
            "fail", "failed", "pass", "performance", "exam", "result", "backlog",
            "arrear", "struggling", "improve", "study", "revision", "weak",
            "poor", "low", "bad", "how to study", "study tips", "score well",
            "how to get good marks", "exam preparation", "semester exam",
        ],
        "system_prompt": """You are an Academic Performance Coach who has helped thousands of engineering students improve their grades, attendance, and overall academic performance.

STUDY SCIENCE YOU APPLY:
- Active Recall: Testing yourself is 2–3x more effective than re-reading
- Spaced Repetition: Review at expanding intervals (1 day → 3 days → 1 week → 1 month)
- Interleaving: Mix subjects while studying — don't block study same subject for hours
- Pomodoro Technique: 25 min focused work + 5 min break × 4, then long break
- Feynman Technique: Explain the concept in simple words — if you can't, you don't know it
- Cornell Notes: Structured note-taking with summary and questions column

CGPA IMPACT ON OPPORTUNITIES:
- CGPA ≥ 8.5: Eligible for most companies, scholarship applications, IIT MTech
- CGPA 7.0–8.4: Good range, some companies filter at 7.5, compensate with projects
- CGPA 6.0–6.9: Compensate hard with skills, projects, competitive programming
- CGPA < 6.0: Focus on skill-based roles, startups, freelancing — be honest about it
- Note: Many product companies (Google, Microsoft, Amazon) don't have CGPA cutoffs

ATTENDANCE RULES (typical Indian universities):
- 75%: Minimum to write exams (most universities)
- 65–74%: Condonation application needed (medical/valid reason)
- < 65%: Detained — must repeat semester
- Strategy: Track monthly, act early if dipping

EXAM STRATEGY BY TIMEFRAME:
- 4+ weeks out: Cover all topics, solve previous year papers, identify weak areas
- 2–3 weeks: Intensive revision, solve model papers, focus on high-weightage units
- 1 week: Quick revision of formulas/definitions, attempt full mock exams
- Day before: Light revision only, sleep 7–8 hours, no all-nighters

STUDENT CONTEXT:
{student_context}

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{history}

Student Question: {question}

Give specific, practical improvement strategies. Be empathetic but direct. Include:
1. Root cause analysis of the problem
2. Step-by-step improvement plan with timeline
3. Specific resources and techniques
4. How to recover and prevent the issue from recurring"""
    },

    "mentorship": {
        "emoji": "🧑‍🏫",
        "name": "Mentor & Support",
        "color": "#D85A30",
        "bg": "#FAECE7",
        "keywords": [
            "stress", "anxious", "anxiety", "motivation", "confidence",
            "depressed", "overwhelmed", "tired", "burnout", "focus",
            "distracted", "personal", "family", "mental", "health",
            "balance", "help me", "lonely", "pressure", "worried",
            "afraid", "scared", "lost", "confused", "don't know",
            "what should i do", "feeling", "feel", "demotivated",
            "give up", "quit", "dropout", "worthless", "comparison",
        ],
        "system_prompt": """You are a compassionate Academic Mentor and Life Coach who genuinely cares about student wellbeing. You combine emotional intelligence with practical wisdom.

YOUR APPROACH:
1. Always acknowledge feelings first — before giving any advice
2. Normalize the struggle — every successful person faced the same doubts
3. Separate the problem from the person — they are not their grades or failures
4. Offer concrete coping tools, not just "stay positive" platitudes
5. Know when to refer to professional support

COMMON STUDENT STRUGGLES YOU HANDLE:
- Exam anxiety and performance pressure from family
- Comparison with peers and social media pressure
- Feeling behind or like an "imposter"
- Homesickness and adjustment to college life
- Career confusion and uncertainty about the future
- Burnout from juggling academics, projects, and extracurriculars
- Relationship issues affecting academic performance
- Financial stress and worries about future employment

PRACTICAL TOOLS YOU RECOMMEND:
- Journaling: 5 minutes daily — write what's working, what's not, what you're grateful for
- The 2-Minute Rule: If a task takes < 2 min, do it now
- Body scan meditation: 10 min daily for anxiety (free on YouTube)
- Weekly review: Every Sunday, plan the week ahead in writing
- Social support: Identify 2–3 people you can talk to honestly
- Digital detox: Set phone-free study blocks and sleep without phone

PROFESSIONAL RESOURCES (India):
- iCall (TISS): 9152987821 — free psychological counseling
- Vandrevala Foundation: 1860-2662-345 (24/7)
- College counseling center — most institutions have one, use it without stigma
- The Live Love Laugh Foundation: for depression awareness

STUDENT CONTEXT:
{student_context}

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{history}

Student's Message: {question}

Respond with warmth and genuine empathy FIRST. Then offer practical support. Never dismiss or minimize their feelings. If they mention self-harm or crisis, prioritize professional resources immediately."""
    },

    "administrative": {
        "emoji": "📝",
        "name": "Administrative Help",
        "color": "#993556",
        "bg": "#FBEAF0",
        "keywords": [
            "register", "registration", "form", "fee", "deadline",
            "policy", "rule", "regulation", "exam form", "hall ticket",
            "bonafide", "certificate", "transfer", "migration", "hostel",
            "scholarship", "document", "application", "procedure", "office",
            "re-evaluation", "revaluation", "supplementary", "backlog exam",
            "noc", "no objection certificate", "tc", "migration certificate",
            "lateral entry", "credit transfer", "leave",
        ],
        "system_prompt": """You are an experienced Academic Administrative Officer who knows all college processes, forms, deadlines, and regulations inside out.

COMMON PROCESSES YOU HANDLE:
1. Course Registration:
   - Login to student portal → Academics → Course Registration
   - Add core subjects first (mandatory), then electives
   - Verify registered courses before deadline
   - Add/Drop period: usually first 2–3 weeks of semester

2. Examination Forms:
   - Fill 30 days before exam date (varies by university)
   - Verify all subjects match registered courses
   - Pay exam fee before submission deadline
   - Download hall ticket 1 week before exam

3. Scholarship Applications:
   - Merit Scholarship: CGPA 8.5+ usually required; apply in July–August
   - Government Scholarships: OBC/SC/ST/EBC → National Scholarship Portal (NSP)
   - Sports Quota Benefits: Apply to Sports Office with achievement certificates
   - Need-Based Aid: Financial Aid Office with income proof (Form 16 / salary slips)
   - Minority Scholarships: Maulana Azad / Begum Hazrat Mahal scholarship portals

4. Certificates & Documents:
   - Bonafide Certificate: Apply 3 working days in advance; used for bank, visa, etc.
   - Character Certificate: Apply to principal office; takes 5–7 days
   - Migration Certificate: Apply 15 days before needed; fee ~₹500; required for transfer
   - Transcript: Official academic record; apostille available for international use

5. Academic Grievances:
   - Grade dispute: → Department HOD → Academic Dean → Controller of Examinations
   - Re-evaluation: Apply within 2 weeks of result; fee ~₹300–500 per subject
   - Attendance condonation: Medical certificate + application to HOD/Dean

STUDENT CONTEXT:
{student_context}

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{history}

Student Question: {question}

Provide precise, step-by-step guidance. Be specific about:
1. The exact steps to follow
2. Documents required
3. Deadlines and fees if applicable
4. Who to contact if there are complications"""
    },
}


# ─────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────

class SimpleHashEmbeddings(Embeddings):
    """Deterministic 1536-d char n-gram embeddings. Replace with OpenAI in production."""
    def __init__(self, dim: int = EMBED_DIMENSION):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        text = text.lower()[:2000]
        vec = [0.0] * self.dim
        for i in range(len(text) - 2):
            h = int(hashlib.md5(text[i:i+3].encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


# ─────────────────────────────────────────────────────────────────
# Web Scraper
# ─────────────────────────────────────────────────────────────────

def scrape_url(url: str, category: str = "general", timeout: int = 15) -> Optional[Document]:
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
        return None

    print(f"  ✓  Scraped [{category}] '{title[:60]}' ({len(raw_text):,} chars)")
    return Document(
        page_content=raw_text,
        metadata={"source": url, "title": title, "category": category,
                  "scraped_at": datetime.now().isoformat()}
    )


# ─────────────────────────────────────────────────────────────────
# Role Detector  (expanded + weighted)
# ─────────────────────────────────────────────────────────────────

def detect_role(query: str, profile: "StudentProfile") -> str:
    """
    Score each role using keyword matching + profile context.
    Returns the best-fit role ID.
    """
    q = query.lower()
    scores: Dict[str, float] = {r: 0.0 for r in ROLES}

    for role_id, role in ROLES.items():
        for kw in role["keywords"]:
            if kw in q:
                # longer keyword matches score higher (more specific)
                scores[role_id] += 1 + len(kw.split()) * 0.5

    # Boost job_market for questions about "which field / technology is better"
    if any(w in q for w in ["vs", "versus", "better", "trending", "demand", "market", "2025"]):
        scores["job_market"] += 2

    # Boost higher_studies for specific exam names
    if any(w in q for w in ["gate", "gre", "gmat", "ielts", "toefl", "nus", "tum", "cmu"]):
        scores["higher_studies"] += 3

    # Boost career for salary / company questions
    if any(w in q for w in ["salary", "lpa", "ctc", "package", "offer", "placed", "placement"]):
        scores["career_planning"] += 2

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "academic_guidance"


# ─────────────────────────────────────────────────────────────────
# Student Profile
# ─────────────────────────────────────────────────────────────────

class StudentProfile:
    def __init__(self, student_id: str):
        self.student_id  = student_id
        self.name        = None
        self.semester    = None
        self.department  = None
        self.cgpa        = None
        self.attendance  = None
        self.goals       = []
        self.concerns    = []
        self.courses     = []   # enrolled courses
        self.session_log : List[Dict] = []

    @staticmethod
    def _normalize_course(course: object) -> Dict[str, object]:
        if isinstance(course, str):
            return {
                "name": course.strip(),
                "code": course.strip().upper(),
                "credits": 0,
                "type": "core"
            }

        if not isinstance(course, dict):
            raise ValueError("Course must be a dict or string")

        code = str(course.get("code") or course.get("course_code") or "").strip().upper()
        name = str(course.get("name") or course.get("title") or code).strip()
        credits = course.get("credits", 0)
        try:
            credits = int(credits)
        except (ValueError, TypeError):
            credits = 0
        course_type = str(course.get("type") or course.get("category") or "core").strip().lower()
        if course_type not in ["core", "elective", "other"]:
            course_type = "core"

        return {
            "name": name,
            "code": code,
            "credits": credits,
            "type": course_type,
        }

    def has_course(self, course_code: str) -> bool:
        code = str(course_code).strip().upper()
        return any(
            isinstance(c, dict) and c.get("code", "").upper() == code
            for c in self.courses
        )

    def add_course(self, course: object) -> None:
        normalized = self._normalize_course(course)
        if not normalized["code"] or not normalized["name"]:
            raise ValueError("Course name and code are required")
        if self.has_course(normalized["code"]):
            raise ValueError(f"Course {normalized['code']} is already added")
        self.courses.append(normalized)

    def remove_course(self, course_code: str) -> None:
        code = str(course_code).strip().upper()
        self.courses = [
            c for c in self.courses
            if not (isinstance(c, dict) and c.get("code", "").upper() == code)
        ]

    def total_credits(self) -> int:
        return sum(
            int(c.get("credits", 0))
            for c in self.courses
            if isinstance(c, dict)
        )

    def electives_count(self) -> int:
        return sum(
            1 for c in self.courses
            if isinstance(c, dict) and c.get("type", "").lower() == "elective"
        )

    def update_from_query(self, query: str):
        """Auto-extract profile hints from natural conversation."""
        q = query.lower()

        # Semester detection
        for pattern in [r"semester\s*(\d)", r"sem\s*(\d)", r"(\d)(?:st|nd|rd|th)?\s*sem"]:
            m = re.search(pattern, q)
            if m:
                sem = int(m.group(1))
                if 1 <= sem <= 8:
                    self.semester = sem

        # CGPA detection
        m = re.search(r"cgpa\s*(?:is|of|:)?\s*(\d+\.\d+)", q)
        if m:
            self.cgpa = float(m.group(1))

        # Attendance
        m = re.search(r"(\d+)\s*%?\s*(?:attendance)", q)
        if m:
            self.attendance = int(m.group(1))

        # Department
        for dept in ["cse", "ece", "it", "mech", "civil", "eee", "mca", "mba"]:
            if dept in q:
                self.department = dept.upper()

    def log(self, role: str, query: str, answer: str):
        self.session_log.append({
            "timestamp": datetime.now().strftime("%H:%M"),
            "role": role,
            "query": query,
            "answer_snippet": answer[:150] + "…" if len(answer) > 150 else answer,
        })

    def context_string(self) -> str:
        parts = []
        if self.name:        parts.append(f"Name: {self.name}")
        if self.department:  parts.append(f"Department: {self.department}")
        if self.semester:    parts.append(f"Current Semester: {self.semester}")
        if self.cgpa:        parts.append(f"CGPA: {self.cgpa}")
        if self.attendance:  parts.append(f"Attendance: {self.attendance}%")
        if self.goals:       parts.append(f"Career Goals: {', '.join(self.goals)}")
        if self.courses:
            course_list = []
            for c in self.courses:
                if isinstance(c, dict):
                    course_list.append(f"{c.get('name')} ({c.get('code')})")
                else:
                    course_list.append(str(c))
            parts.append(f"Enrolled Courses: {', '.join(course_list)}")
        if not parts:
            return "No profile information collected yet."
        return "\n".join(parts)

    def summary(self) -> str:
        return self.context_string()


# ─────────────────────────────────────────────────────────────────
# Pinecone helper
# ─────────────────────────────────────────────────────────────────

def ensure_index(pc: Pinecone, name: str, dim: int):
    existing = [i.name for i in pc.list_indexes()]
    if name not in existing:
        print(f"  Creating Pinecone index '{name}' …")
        pc.create_index(
            name=name, dimension=dim, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(2)
        print("  ✓  Index ready.")
    else:
        print(f"  ✓  Using existing index '{name}'.")


# ─────────────────────────────────────────────────────────────────
# Main Agent
# ─────────────────────────────────────────────────────────────────

class AcademicAdvisorAgent:
    """
    Upgraded Academic Advisor with:
    - 7 specialist roles (added Job Market + Higher Studies)
    - Claude-first answering with RAG as enriching context
    - Full conversation memory per student
    - Smart role detection with profile awareness
    - Structured, markdown-formatted responses
    """

    def __init__(self):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set in .env")

        print("\n🎓 Initialising Academic Advisor Agent v2 …")

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        self.model = OPENROUTER_MODEL

        self.embeddings = SimpleHashEmbeddings(EMBED_DIMENSION)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=self.embeddings,
        )

        # Per-student memory and profiles
        self.profiles : Dict[str, StudentProfile] = {}
        self.memories : Dict[str, SimpleMemory] = {}

        print("✅ Academic Advisor v2 ready!\n")
        for r in ROLES.values():
            print(f"    {r['emoji']}  {r['name']}")
        print()

    def _ensure_profile(self, student_id: str) -> StudentProfile:
        if student_id not in self.profiles:
            self.profiles[student_id] = StudentProfile(student_id)
        return self.profiles[student_id]

    def get_courses(self, student_id: str) -> List[Dict[str, object]]:
        profile = self._ensure_profile(student_id)
        return [c for c in profile.courses]

    def add_course(self, student_id: str, course_data: dict) -> List[Dict[str, object]]:
        profile = self._ensure_profile(student_id)
        profile.add_course(course_data)
        return self.get_courses(student_id)

    def remove_course(self, student_id: str, course_code: str) -> List[Dict[str, object]]:
        profile = self._ensure_profile(student_id)
        profile.remove_course(course_code)
        return self.get_courses(student_id)

    def course_summary(self, student_id: str) -> Dict[str, object]:
        profile = self._ensure_profile(student_id)
        return {
            "courses": [c for c in profile.courses],
            "total_credits": profile.total_credits(),
            "enrolled_count": len(profile.courses),
            "electives_count": profile.electives_count(),
        }

    # ── Knowledge ingestion ──────────────────────────────────────

    def ingest_urls(self, urls: List[str], category: str = "general") -> int:
        print(f"\n📥 Ingesting {len(urls)} URL(s) [{category}] …")
        docs = []
        for url in urls:
            doc = scrape_url(url, category=category)
            if doc:
                docs.append(doc)
            time.sleep(0.5)
        if not docs:
            return 0
        chunks = self.splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)
        print(f"  📌 Stored {len(chunks)} chunks.\n")
        return len(chunks)

    def ingest_text(self, text: str, title: str, category: str = "general") -> int:
        doc = Document(
            page_content=text,
            metadata={"source": "manual", "title": title, "category": category}
        )
        chunks = self.splitter.split_documents([doc])
        self.vectorstore.add_documents(chunks)
        print(f"  📌 Ingested '{title}' → {len(chunks)} chunks [{category}]")
        return len(chunks)

    # ── Core advise method ───────────────────────────────────────

    def advise(self, query: str, student_id: str = "guest",
               force_role: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point. Detects role → retrieves RAG context →
        builds rich prompt → Claude generates meaningful answer.
        """
        # Get/create student profile and memory
        if student_id not in self.profiles:
            self.profiles[student_id] = StudentProfile(student_id)
        if student_id not in self.memories:
            self.memories[student_id] = SimpleMemory(k=8)

        profile = self.profiles[student_id]
        memory  = self.memories[student_id]

        # Auto-extract profile hints from the query
        profile.update_from_query(query)

        # Detect role
        role_id = (force_role if force_role and force_role in ROLES
                   else detect_role(query, profile))
        role = ROLES[role_id]

        # Retrieve relevant context from vector store
        rag_context = self._retrieve_context(query)

        # Build conversation history string
        history = self._format_history(memory)

        # Build the full system prompt
        system_prompt = role["system_prompt"].format(
            student_context=profile.context_string(),
            rag_context=rag_context,
            history=history,
            question=query,
        )

        # Call OpenRouter
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        answer = self._call_llm(messages)

        # Save to memory
        memory.save_context({"input": query}, {"output": answer})
        profile.log(role_id, query, answer)

        # Gather sources
        sources = self._get_sources(query)

        return {
            "answer"     : answer,
            "role"       : role["name"],
            "role_emoji" : role["emoji"],
            "role_id"    : role_id,
            "role_color" : role["color"],
            "sources"    : sources,
            "student_id" : student_id,
            "timestamp"  : datetime.now().strftime("%H:%M"),
        }

    def _call_llm(self, messages: List[Any]) -> str:
        """Helper to call OpenRouter via OpenAI client."""
        payload = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            else:
                continue
            payload.append({"role": role, "content": m.content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def _retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve top-k relevant chunks from Chroma."""
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            if not docs:
                return "No specific knowledge base context found — answer from expert knowledge."
            return "\n\n---\n\n".join(
                f"[Source: {d.metadata.get('title','Unknown')}]\n{d.page_content}"
                for d in docs
            )
        except Exception:
            return "Knowledge base unavailable — answering from expert knowledge."

    def _get_sources(self, query: str) -> List[str]:
        try:
            docs = self.vectorstore.similarity_search(query, k=3)
            return list({d.metadata.get("source", "internal") for d in docs})
        except Exception:
            return []

    def _format_history(self, memory: SimpleMemory) -> str:
        try:
            msgs = memory.chat_memory.messages
            if not msgs:
                return "No previous conversation in this session."
            parts = []
            for m in msgs[-6:]:   # last 3 exchanges
                role = "Student" if isinstance(m, HumanMessage) else "Advisor"
                parts.append(f"{role}: {m.content[:200]}{'…' if len(m.content)>200 else ''}")
            return "\n".join(parts)
        except Exception:
            return ""

    # ── Progress report ──────────────────────────────────────────

    def generate_progress_report(self, student_id: str) -> str:
        if student_id not in self.profiles:
            return "No profile found for this student."
        profile = self.profiles[student_id]
        log_text = "\n".join(
            f"[{e['timestamp']}] {e['role'].upper()}: {e['query'][:80]} → {e['answer_snippet']}"
            for e in profile.session_log
        ) or "No interactions yet."

        return self._call_llm([
            SystemMessage(content="""You are an academic advisor generating a detailed student progress report.
Be specific, constructive, and encouraging. Identify patterns, highlight strengths, flag concerns, recommend next actions."""),
            HumanMessage(content=f"""Generate a comprehensive academic progress report for:

STUDENT PROFILE:
{profile.summary()}

SESSION INTERACTIONS:
{log_text}

Structure the report as:
1. Profile Summary
2. Topics the student is focused on
3. Strengths identified
4. Areas needing attention
5. Recommended next 3 actions (specific and time-bound)
6. Motivational closing message""")
        ])

    def stats(self) -> dict:
        return {
            "pinecone_stats"  : self.pinecone_index.describe_index_stats(),
            "active_profiles" : len(self.profiles),
            "roles"           : {k: v["name"] for k, v in ROLES.items()},
        }


# ─────────────────────────────────────────────────────────────────
# Built-in Knowledge Base
# ─────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = {

    "academic_guidance": """
ENGINEERING DEGREE STRUCTURE (B.Tech / B.E. — 4 Years):
Semester 1–2: Foundation — Mathematics I & II, Physics, Chemistry, Programming, Engineering Drawing
Semester 3–4: Core subjects — Data Structures, Algorithms, Digital Electronics, Signals, DBMS, OOP
Semester 5–6: Specialization begins — OS, CN, Compiler Design + Electives start
Semester 7–8: Advanced electives + Final Year Project (mandatory, 12 credits)
Total credits: 160–180 depending on university

ELECTIVE SELECTION STRATEGY:
- Pick electives that cluster around ONE specialization area (ML, Web, Embedded, Security)
- Don't pick electives based on "easy" — pick based on career alignment
- Maximum recommended: 2–3 electives per semester alongside 4 core subjects
- Prerequisite awareness: ML requires Probability+Stats+Linear Algebra base

CGPA STRATEGY:
- CGPA ≥ 8.5: Opens doors to top companies + IIT MTech + scholarships
- CGPA 7.0–8.4: Compensate with strong projects and internships
- CGPA < 7.0: Focus on skill certifications, competitive programming, startups
- First 4 semesters matter most for cumulative calculation — protect early GPA

SEMESTER-WISE COURSE LOAD ADVICE:
Sem 5 CSE recommended: Machine Learning, Database Management, Computer Networks, OS, + 1 Elective
Sem 6 CSE recommended: Deep Learning / NLP, Compiler Design, Cloud Computing, Software Engineering, + 1 Elective
Sem 7 CSE recommended: Project-heavy semester + 2 Advanced Electives + Internship component
Sem 8 CSE recommended: Final Year Project (6 months, full focus) + 1–2 light electives
""",

    "career_planning": """
CAREER PATHS FOR CSE/IT ENGINEERS:

1. SOFTWARE DEVELOPMENT
Entry roles: SDE-1, Junior Developer, Associate Engineer
Skills: DSA, System Design, 1–2 languages (Python/Java/Go), Git, SQL
Companies: Amazon, Microsoft, Flipkart, Swiggy, TCS, Infosys, etc.
Salary range: ₹3.5 LPA (services) to ₹35 LPA (top product cos)

2. AI / MACHINE LEARNING
Entry roles: ML Engineer, Data Scientist, AI Engineer
Skills: Python, NumPy/Pandas, scikit-learn, PyTorch/TensorFlow, SQL, Statistics
Companies: Google, Microsoft AI, Amazon Science, Flipkart AI, Zepto, startups
Salary range: ₹10–28 LPA fresher in product cos

3. DATA ENGINEERING
Entry roles: Data Engineer, Analytics Engineer, BI Developer
Skills: SQL, Spark, Airflow, dbt, Kafka, Python, Cloud (AWS/GCP)
Companies: Swiggy, Zomato, Juspay, Groww, Chargebee, Adobe
Salary range: ₹8–22 LPA — less competition than Data Science

4. CLOUD / DEVOPS
Entry roles: Cloud Engineer, DevOps Engineer, SRE
Skills: AWS/GCP/Azure, Docker, Kubernetes, Terraform, CI/CD, Linux
Salary range: ₹8–25 LPA — AWS certification alone worth ₹3–5 LPA premium

5. PRODUCT MANAGEMENT
Entry roles: Associate PM, Technical PM (requires 2–3 years SDE first usually)
Skills: SQL, user research, wireframing, product sense, communication
Companies: Flipkart, Meesho, PhonePe, CRED, Razorpay
Salary: ₹20–50 LPA for experienced PMs

INTERNSHIP STRATEGY:
Apply 5–6 months before start date
Platforms: LinkedIn, Internshala, AngelList, LeetCode Jobs, company career pages
Referrals are 5x more effective than cold applications — network actively
Target: 1 strong internship in Sem 5–6, convert to PPO if possible
PPO (Pre-Placement Offer) converts ~40–60% at product companies

PLACEMENT PREPARATION TIMELINE (for Sem 8 placements):
Sem 5: Start competitive programming (LeetCode), build first ML/web project
Sem 6: 150+ LeetCode problems, internship, GitHub with 3+ projects
Sem 7: Complete 300+ LeetCode, system design prep, mock interviews
Sem 8 (Aug–Oct): Peak placement season — tests, interviews, offers
""",

    "job_market": """
JOB MARKET TRENDS 2024–2025 FOR ENGINEERING GRADUATES:

HOT SKILLS RIGHT NOW:
1. Generative AI / LLM Engineering — HIGHEST demand, lowest supply
   Tools: LangChain, LlamaIndex, HuggingFace, OpenAI API, vector databases
   Salary premium: 30–50% above average SDE

2. MLOps / AI Infrastructure
   Tools: MLflow, Weights & Biases, Kubeflow, Sagemaker, Vertex AI
   Growing fast as companies productionize their AI systems

3. Cloud-Native Development
   Tools: AWS/GCP/Azure, Kubernetes, Terraform, Serverless
   AWS/GCP certifications: 15–25% salary premium

4. Full Stack (React + Python/Node)
   Always in demand, moderate competition, good entry point

5. Data Engineering
   Tools: Apache Spark, Kafka, Airflow, dbt, Snowflake
   Less competition than Data Science, comparable or better salaries

6. Cybersecurity
   Growing due to regulations (DPDP Act in India, GDPR globally)
   Certifications matter: CEH, CISSP, CompTIA Security+

COMPANIES ACTIVELY HIRING FRESHERS (2025):
Tier 1 Product: Google, Microsoft, Amazon, Adobe, Salesforce, Atlassian
Indian Product: Flipkart, Zomato, Swiggy, Zepto, CRED, Razorpay, PhonePe, Groww
GCCs (pay like MNCs): JP Morgan Tech, Goldman Sachs Engineering, Deutsche Bank Tech
Mid-product: Chargebee, Juspay, Hasura, BrowserStack, Postman, Setu
Services (mass hiring): TCS, Infosys, Wipro, Cognizant, Capgemini, HCL

HONEST MARKET REALITY:
- 2022 boom → 2023–24 correction → 2025 recovery (AI-driven)
- Layoffs were at senior levels — fresher hiring continued at most product cos
- Service company packages: ₹3.5–6 LPA (but skill development opportunities exist)
- The gap between skilled and unskilled fresh engineers is WIDENING — upskill aggressively

REMOTE WORK STATUS:
- Pure remote: Rare for freshers in India (30% reduction from 2022 peak)
- Hybrid (3 days office): Standard at most product cos
- Full remote: Available at some global-remote startups and freelance

INDIA VS. ABROAD JOB COMPARISON:
India (top product): ₹20–40 LPA fresher
USA (top tech): $130–180K USD = ₹110–150 LPA (but higher cost of living)
Germany: €50–70K = ₹45–60 LPA (lower living cost, work-life balance)
Singapore: SGD 70–100K = ₹43–62 LPA (gateway to APAC)
Canada: CAD 80–100K = ₹48–60 LPA (PR pathway advantage)
""",

    "higher_studies": """
HIGHER STUDIES GUIDE FOR ENGINEERING STUDENTS:

GATE EXAM (DOMESTIC MASTERS):
- Papers: CS/IT, EC, ME, CE, EE, etc.
- Score needed: 750–850 for IIT (top 5), 700+ for remaining IITs, 600+ for NITs
- Validity: 3 years
- Benefit: MTech stipend ₹12,400–₹37,000/month + IIT brand
- PSU recruitment: BHEL, ONGC, IOCL, PGCIL recruit via GATE score
- Timeline: Exam in Feb; apply for colleges April–May; start prep 12–18 months early
- Resources: NPTEL lectures (free), Made Easy/GATEOverflow for CS

MS ABROAD — USA:
- Top programs: CMU MCDS/MSML, Stanford MS CS, MIT EECS, UC Berkeley EECS
- Safe targets with funding: UMass Amherst, UT Dallas, SUNY Buffalo, IIT Chicago
- Requirements: GRE 320+ (many waived), TOEFL 100+, CGPA 8.0+ preferred
- MS cost: $25–45K/year; 1.5–2 year program; total ₹40–70 lakhs
- ROI: US job $120–150K+ makes it worth it; break even in 2–3 years
- Research MS vs. Coursework MS: Research MS → better for PhD/research roles
- Important: Email professors for research fit — can lead to RA/TA funding

MS ABROAD — GERMANY (Affordable Option):
- TU Munich, RWTH Aachen, KIT, TU Berlin — world-class programs
- Cost: Near-zero tuition (€250–350/semester admin fee), living ~€800–1000/month
- Programs in English: Informatics at TUM, Data Engineering at FAU
- Requirements: CGPA 8.0+, IELTS 6.5+ or English-taught B.Sc proof
- Timeline: Apply Aug–Oct for winter semester (Jan), Jan–Apr for summer (April)
- Advantage: Easier PR/work visa after graduation

PHD:
- Fully funded at top US/European universities (stipend $20–35K/year + tuition)
- Requirements: Strong research background, publications preferred, excellent LORs
- Best for: Those passionate about research and academia / R&D industry roles
- Timeline: 4–6 years; career outcomes: academia, research labs (Google Brain, DeepMind, MSR)

MBA:
- IIM (CAT): CAT 99%ile for top IIMs (A/B/C); 95%ile for rest
- ISB Hyderabad: GMAT 710+; expensive (₹40L) but excellent ROI
- Global MBA: Harvard, Wharton, INSEAD — GMAT 720+, 3–5 years work experience needed
- Good for: Transition to consulting, product management, entrepreneurship
- Not needed for: Technical IC roles (software, ML, data)
""",

    "mentorship_wellness": """
STUDENT WELLNESS GUIDE:

MANAGING EXAM STRESS:
- Stress is normal — it means you care. The goal is optimal arousal, not zero stress.
- Physical: 30 min walk daily reduces cortisol by 26%; sleep 7–8 hours non-negotiably
- Mental: Break exam prep into daily small goals — the mountain feels smaller
- Social: Study with 1–2 friends for accountability without distraction

COMMON TRAPS TO AVOID:
- All-nighters: Sleep deprivation reduces memory consolidation by 40% — don't do it
- Comparison: Everyone's journey is different; comparison is the thief of joy
- Social media during exams: Delete or mute for 2 weeks before finals
- Perfectionism paralysis: Done > perfect. Start, then improve.

TIME MANAGEMENT SYSTEM:
1. Sunday planning ritual: List everything due in the week, assign to daily slots
2. Time blocking: Dedicate specific hours to each subject/project
3. Energy matching: Hard tasks in peak hours, admin tasks when tired
4. Buffer time: Leave 30% of your schedule empty for the unexpected

BUILDING RESILIENCE:
- Failure is data: Every failure tells you what to do differently
- Growth mindset: "I can't do this YET" vs "I can't do this"
- Support network: Build 2–3 genuine relationships in college
- Mentors: Find a senior or professor who can guide you beyond academics

PROFESSIONAL HELP RESOURCES (India):
- iCall (TISS): 9152987821 — free, confidential psychological support
- Vandrevala Foundation: 1860-2662-345 (24/7 helpline)
- The Live Love Laugh Foundation: depression and mental health resources
- Your college counseling center — use it, there is no stigma in seeking help

MOTIVATION WHEN FEELING LOST:
- It's normal to not have everything figured out in college
- Most successful engineers changed direction multiple times
- Focus on 1 next step, not the whole journey
- Track progress weekly — you are growing even when it doesn't feel like it
""",

    "administrative": """
ACADEMIC ADMINISTRATION COMPLETE GUIDE:

COURSE REGISTRATION:
Step 1: Login to Student Portal (ERP/LMS)
Step 2: Navigate to Academics → Semester Registration
Step 3: View your credit eligibility (based on previous semester CGPA and backlog status)
Step 4: Add core subjects (mandatory first) → then electives from approved list
Step 5: Verify total credits are within limit (usually 22–26 max)
Step 6: Submit before deadline (typically Week 1–2 of semester)
Step 7: Confirm in "My Enrolled Courses" tab
Late registration: Fine of ₹100–500/day depending on university

EXAM FORM PROCEDURE:
- Fill via student portal → Examination → Semester Exam Form
- Verify all registered subjects appear; add arrear subjects if applicable
- Pay exam fee before portal closes
- Download and print hall ticket 5 days before exam
- Carry original ID + hall ticket to every exam

IMPORTANT DOCUMENTS AND HOW TO GET THEM:
Bonafide Certificate: Portal → Certificates → Bonafide; takes 2–3 working days; needed for bank, passport, visa
Character Certificate: Apply to principal office; takes 5–7 days
Official Transcript: Academic section office; apostille required for foreign universities; ₹500–1000
Migration Certificate: Apply 15 days before needed; required for university transfer; ₹500
TC (Transfer Certificate): Required when leaving college permanently

SCHOLARSHIP DEADLINES AND ELIGIBILITY:
Merit Scholarship: CGPA 8.5+; apply June–August each year
National Scholarship Portal (NSP): OBC/SC/ST/EBC; apply September–October
Post-Matric Scholarship: State government portal; apply within 3 months of admission
Minority Scholarships: Maulana Azad and Begum Hazrat Mahal portals; apply October
Central Sector Scholarship: Top 20th percentile in Class 12 board; renewal each year

ACADEMIC GRIEVANCE PROCESS:
Grade dispute: Class teacher → HOD → Academic Dean → Controller of Examinations
Re-evaluation request: Apply within 2 weeks of result publication; ₹300–500/subject
Attendance condonation: Medical certificate → HOD → Dean of Students
Examination malpractice appeal: Examination committee → Academic council
"""
}


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║      🎓  ACADEMIC ADVISOR  v2  —  Powered by Claude + RAG       ║
║   7 roles · Live market data · Memory · Meaningful answers      ║
╚══════════════════════════════════════════════════════════════════╝

Ask anything about:
  📚 Courses & curriculum      🧭 Career paths & companies
  📈 Job market & salaries     🎯 Higher studies (GATE/GRE/MS/PhD)
  📊 Grades & study tips       🧑‍🏫 Stress & motivation
  📝 College admin & forms

Commands:  report | profile | stats | role:<id> <question> | quit
"""

def main():
    agent = AcademicAdvisorAgent()

    print("\n📚 Loading knowledge base …")
    for category, text in KNOWLEDGE_BASE.items():
        label = category.replace("_", " ").title()
        agent.ingest_text(text, title=f"Built-in: {label}", category=category)
    print()

    print(BANNER)

    try:
        student_id = input("Student ID (press Enter for 'guest'): ").strip() or "guest"
        name = input("Your name (optional): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!"); return

    profile = agent.profiles.setdefault(student_id, StudentProfile(student_id))
    if name:
        profile.name = name

    print(f"\n👋 Welcome{' ' + name if name else ''}! Ask me anything academic.\n")

    while True:
        try:
            raw = input(f"[{student_id}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye! All the best with your studies!"); break

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("quit", "exit", "q"):
            print("👋 Goodbye!"); break

        elif cmd == "profile":
            print(f"\n👤 Profile:\n{profile.summary()}")
            print(f"   Interactions: {len(profile.session_log)}")

        elif cmd == "report":
            print("\n⏳ Generating progress report …\n")
            print("─" * 60)
            print(agent.generate_progress_report(student_id))
            print("─" * 60)

        elif cmd == "stats":
            s = agent.stats()
            print(f"\n📊 Active profiles: {s['active_profiles']}")
            print(f"   Roles: {', '.join(s['roles'].values())}")

        elif cmd == "help" or cmd == "?":
            print(BANNER)

        elif raw.lower().startswith("role:"):
            parts = raw[5:].split(" ", 1)
            if len(parts) == 2:
                resp = agent.advise(parts[1].strip(), student_id=student_id,
                                    force_role=parts[0].strip())
                _print_response(resp)
            else:
                print("Usage: role:<role_id> <question>")
                print(f"Roles: {', '.join(ROLES.keys())}")

        else:
            resp = agent.advise(raw, student_id=student_id)
            _print_response(resp)


def _print_response(resp: Dict):
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"{resp['role_emoji']}  {resp['role']}  [{resp['timestamp']}]")
    print(sep)
    print(resp["answer"])
    if resp["sources"] and "manual" not in resp["sources"]:
        print(f"\n📚 Sources: {', '.join(resp['sources'][:3])}")
    print(sep + "\n")


if __name__ == "__main__":
    main()
