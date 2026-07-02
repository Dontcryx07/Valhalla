""" File containing configuration variables """

from pathlib import Path

# backend/src/config.py -> backend/src -> backend -> Valhalla (project root)
BASE_DIR = Path(__file__).resolve().parents[2]

BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BACKEND_DIR / "data"

ENVIRONMENT_DIR = DATA_DIR / "environment"
PERSONALITIES_DIR = DATA_DIR / "personalities"

OUTPUT_DIR = BACKEND_DIR / "output"

FRONTEND_DIR = BASE_DIR / "frontend"

PLACES_FILE = ENVIRONMENT_DIR / "places.json"


if __name__ == '__main__':
    print(f"Running this project from : {BASE_DIR}")
