""" File containing configuration variables """

from pathlib import Path
import os

# backend/src/config.py -> backend/src -> backend -> Valhalla (project root)
BASE_DIR = Path(__file__).resolve().parents[2]

BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BACKEND_DIR / "data"

ENVIRONMENT_DIR = DATA_DIR / "environment"
PERSONALITIES_DIR = DATA_DIR / "personalities"

OUTPUT_DIR = BACKEND_DIR / "output"

FRONTEND_DIR = BASE_DIR / "frontend"

PLACES_FILE = ENVIRONMENT_DIR / "places.json"

# LLM Configuration
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = "AIzaSyAiRKwlsJNL4rglp1yWJVnmxLp0t_Igsls"


# day_planner.py CONFIG
MAX_PLAN_RETRIES = 3


if __name__ == '__main__':
    print(f"Running this project from : {BASE_DIR}")
