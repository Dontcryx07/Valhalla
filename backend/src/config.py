"""
Project-wide path configuration.
Resolves the project root, backend, frontend, data, and output directories
so all modules can reference consistent paths.
"""

from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path: str | Path) -> bool:
        env_path = Path(path)
        if not env_path.exists():
            return False

        loaded = False
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded = True

        return loaded


# DEBUG -
VERBOSE = True

# backend/src/config.py -> backend/src -> backend -> Valhalla (project root)
BASE_DIR = Path(__file__).resolve().parents[2]

# Load the project-local .env so environment variables are available before
# any config values are read.
load_dotenv(BASE_DIR / ".env")

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

# Time ratios — tweak these to find the simulation's sweet spot.
#   SIM_MINUTES_PER_TICK  = how many simulation minutes advance per engine tick.
#   REAL_SECONDS_PER_SIM_MINUTE = how many wall-clock seconds correspond to
#                                  1 simulation minute for LLM latency budget.
#   REAL_SECONDS_PER_TICK = SIM_MINUTES_PER_TICK * REAL_SECONDS_PER_SIM_MINUTE
#                           (derived convenience constant — do NOT set manually).
SIM_MINUTES_PER_TICK = 1
REAL_SECONDS_PER_SIM_MINUTE = 1.0  # 1 sim-minute = 1 real second (1:1 real time)
REAL_SECONDS_PER_TICK: float = SIM_MINUTES_PER_TICK * REAL_SECONDS_PER_SIM_MINUTE


# LLM Configuration
TEMPERATURE = 0.7           # Creativity the model is allowed
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY_1", "")
GEMINI_MODEL = "gemini-3.5-flash" # Default single model for use

# Fallback chain per task tier. Order = preference order.
MODEL_TIERS: dict[str, list[str]] = {
    # Day-planning, dialogue — uses ONLY gemini-3.5-flash.
    # Rate-limit resilience comes from multiple API keys (see API_KEYS below),
    # not from model fallback.
    "complex": [
        "gemini-3.5-flash",
    ],
    # Cheap / high-volume calls: plan decomposition, small classification.
    "simple": [
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
    ],
}

# Support multiple API keys (comma-separated in env var).
# Rotates through them when rate limits are hit on the current key.
API_KEYS: list[str] = [
    k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()
]
if not API_KEYS:
    # Gather GEMINI_API_KEY (no suffix) + GEMINI_API_KEY_1 through _10
    numbered_keys = [
        os.environ.get(f"GEMINI_API_KEY_{index}", "")
        for index in range(1, 11)
    ]
    API_KEYS = [key for key in [os.environ.get("GEMINI_API_KEY", ""), *numbered_keys] if key]


# Conversation cap (per agent per day)
MAX_CONVERSATIONS_PER_AGENT = 5

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
