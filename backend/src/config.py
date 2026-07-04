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

### Core logic

# Default nearby distance (chebyshev distance)
DEFAULT_PERCEPTION_RADIUS = 5 # Tune after we know the actual map scale


# LLM Configuration
TEMPERATURE = 0.7           # Creativity the model is allowed
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash" # Default single model for use

# Fallback chain per task tier. Order = preference order.
# Put your strongest model first; weaker/cheaper ones behind it as a safety net.
MODEL_TIERS: dict[str, list[str]] = {
    # For day-planning, dialogue, anything needing real reasoning quality.
    "complex": [
        "gemini-3.5-flash",
        "gemini-3.0-flash",
        "gemini-2.5-flash",
    ],
    # For cheap/high-volume calls: plan decomposition, small classification, etc.
    "simple": [
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
        "gemma-4-31B-it",
    ],
}

# Support multiple API keys (comma-separated in env var).
# Rotates through them if a whole model chain gets rate-limited on the current key.
API_KEYS: list[str] = [
    k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()
] or [os.environ.get("GEMINI_API_KEY", "")]


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
