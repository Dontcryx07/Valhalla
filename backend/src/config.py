"""
Project-wide path configuration.
Resolves the project root, backend, frontend, data, and output directories
so all modules can reference consistent paths.
"""

from pathlib import Path
import os


# DEBUG -
VERBOSE = True

# backend/src/config.py -> backend/src -> backend -> Valhalla (project root)
BASE_DIR = Path(__file__).resolve().parents[2]

BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BACKEND_DIR / "data"

ENVIRONMENT_DIR = DATA_DIR / "environment"
PERSONALITIES_DIR = DATA_DIR / "personalities"

OUTPUT_DIR = BACKEND_DIR / "output"

FRONTEND_DIR = BASE_DIR / "frontend"

PLACES_FILE = ENVIRONMENT_DIR / "places.json"


# Logging
LOG_DIR = OUTPUT_DIR / "logs"
LOG_LEVEL = "DEBUG" if VERBOSE else "INFO"


# LLM Configuration
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


# day_planner.py CONFIG
MAX_PLAN_RETRIES = 3
# What is expected in persona.json file
PERSONA_FIELD_GLOSSARY = {
    "daily_plan_req": "A rough sketch of their typical day — classes, work, recurring commitments.",
    "innate": "Personality traits they were simply born with (natural disposition).",
    "learned": "Skills/knowledge acquired since starting college (not innate).",
    "lifestyle": "Habits and routines: sleep schedule, exercise, social patterns.",
    "hobbies": "Free-time activities they actively enjoy.",
    "goals": "Short- and long-term goals, academic and personal.",
}


if __name__ == '__main__':
    print(f"Running this project from : {BASE_DIR}")
    print(GEMINI_API_KEY)
