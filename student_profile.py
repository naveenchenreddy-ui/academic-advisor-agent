"""
StudentProfile — tracks per-student context extracted automatically
from the conversation. Enriches every Claude prompt with student details.
"""
import re
from typing import List, Dict
from datetime import datetime


class StudentProfile:
    def __init__(self, student_id: str):
        self.student_id  = student_id
        self.name        = None
        self.semester    = None
        self.department  = None
        self.cgpa        = None
        self.attendance  = None
        self.goals:    List[str] = []
        self.concerns: List[str] = []
        self.courses:  List[Dict[str, object]] = []
        self.session_log: List[Dict] = []

    def update_from_query(self, query: str):
        """Auto-extract profile hints from natural conversation text."""
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

        # Attendance detection
        m = re.search(r"(\d+)\s*%?\s*(?:attendance)", q)
        if m:
            self.attendance = int(m.group(1))

        # Department detection
        for dept in ["cse", "ece", "it", "mech", "civil", "eee", "mca", "mba"]:
            if dept in q:
                self.department = dept.upper()

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

    def _format_course_label(self, course: Dict[str, object]) -> str:
        return f"{course.get('name')} ({course.get('code')})" if course.get('code') else course.get('name')

    def has_course(self, course_code: str) -> bool:
        code = str(course_code).strip().upper()
        for c in self.courses:
            if isinstance(c, dict) and c.get("code", "").upper() == code:
                return True
        return False

    def add_course(self, course: object) -> None:
        normalized = self._normalize_course(course)
        if not normalized["code"] or not normalized["name"]:
            raise ValueError("Course code and name are required")
        if self.has_course(normalized["code"]):
            raise ValueError(f"Course {normalized['code']} is already added")
        self.courses.append(normalized)

    def remove_course(self, course_code: str) -> None:
        code = str(course_code).strip().upper()
        self.courses = [c for c in self.courses if not (isinstance(c, dict) and c.get("code", "").upper() == code)]

    def course_names(self) -> List[str]:
        return [self._format_course_label(c) if isinstance(c, dict) else str(c) for c in self.courses]

    def total_credits(self) -> int:
        return sum(int(c.get("credits", 0)) for c in self.courses if isinstance(c, dict))

    def electives_count(self) -> int:
        return sum(1 for c in self.courses if isinstance(c, dict) and c.get("type", "").lower() == "elective")

    def context_string(self) -> str:
        """Returns a compact profile context string for injection into prompts."""
        parts = []
        if self.name:        parts.append(f"Name: {self.name}")
        if self.department:  parts.append(f"Department: {self.department}")
        if self.semester:    parts.append(f"Current Semester: {self.semester}")
        if self.cgpa:        parts.append(f"CGPA: {self.cgpa}")
        if self.attendance:  parts.append(f"Attendance: {self.attendance}%")
        if self.goals:       parts.append(f"Career Goals: {', '.join(self.goals)}")
        if self.courses:
            parts.append(f"Enrolled Courses: {', '.join(self.course_names())}")
        return "\n".join(parts) if parts else "No profile information collected yet."

    def summary(self) -> str:
        return self.context_string()

    def log(self, role: str, query: str, answer: str):
        self.session_log.append({
            "timestamp": datetime.now().strftime("%H:%M"),
            "role": role,
            "query": query,
            "answer_snippet": answer[:150] + "…" if len(answer) > 150 else answer,
        })
