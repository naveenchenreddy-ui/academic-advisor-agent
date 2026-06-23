"""
Unit tests for role detection logic.
Does not require API keys — tests only the routing logic.

Usage:
    python tests/test_role_detection.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.student_profile import StudentProfile

TEST_CASES = [
    ("which elective should I pick for ML?",              "academic_guidance"),
    ("what is the salary for ML engineer in India?",      "job_market"),
    ("how to prepare for GATE CS in 6 months?",          "higher_studies"),
    ("I'm really stressed and overwhelmed about exams",   "mentorship"),
    ("how do I fill the exam form and what is deadline?", "administrative"),
    ("my CGPA dropped to 6.2, how to improve?",          "academic_progress"),
    ("best companies for CSE freshers in 2025?",         "job_market"),
    ("should I do MS in USA or Germany for AI?",          "higher_studies"),
    ("what internships should I apply for sem 5?",        "career_planning"),
    ("how many credits do I need to graduate?",           "academic_guidance"),
]


def test_role_detection():
    try:
        from backend.academic_advisor_agent import detect_role
    except ImportError:
        print("Could not import detect_role — check that academic_advisor_agent.py is in backend/")
        return

    profile = StudentProfile("test_student")
    profile.department = "CSE"
    profile.semester = 5
    profile.cgpa = 7.4

    print("Running role detection tests …\n")
    passed = 0
    for query, expected in TEST_CASES:
        detected = detect_role(query, profile)
        ok = detected == expected
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        flag = "" if ok else f"  (expected: {expected})"
        print(f"  [{status}] {query[:55]:<56} → {detected}{flag}")

    print(f"\n{'='*60}")
    print(f"Result: {passed}/{len(TEST_CASES)} passed")
    if passed == len(TEST_CASES):
        print("All tests passed!")
    else:
        print(f"{len(TEST_CASES) - passed} test(s) failed — check role keyword lists.")


if __name__ == "__main__":
    test_role_detection()
