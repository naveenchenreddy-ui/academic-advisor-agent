#!/usr/bin/env python3
"""
Academic Advisor Agent v2 — Entry Point
Run this to start the interactive CLI.

Usage:
    python run.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from academic_advisor_agent import main

if __name__ == "__main__":
    main()
