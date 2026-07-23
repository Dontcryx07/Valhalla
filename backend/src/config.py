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

# ---------------------------------------------------------------------------
# Toggle helpers — read a value from the environment (.env), with a default.
# Every simulation toggle below can be set in .env and overridden on the CLI
# (see apply_overrides() / add_cli_arguments() at the bottom of this file).
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw.strip() if raw not in (None, "") else default


# ---------------------------------------------------------------------------
# Spatial awareness
# ---------------------------------------------------------------------------

# Legacy tile-based radius (chebyshev). Kept for the tile-grid perception path.
DEFAULT_PERCEPTION_RADIUS = _env_int("SIM_PERCEPTION_RADIUS_TILES", 5)

# NEW: pixel radius for the proximity circle used for PERCEPTION (who an agent sees).
PERCEPTION_RADIUS_PX = _env_int("SIM_PERCEPTION_RADIUS_PX", 50)

# Pixel radius for starting a CONVERSATION — agents must be this close to talk
# (tighter than perception, so they see each other before they chat).
CONVERSATION_RADIUS_PX = _env_int("SIM_CONVERSATION_RADIUS_PX", 20)

# Whether the perception step runs each tick (0-LLM either way).
PERCEPTION_ENABLED = _env_bool("SIM_PERCEPTION_ENABLED", True)

# Run perception only every N ticks (1 = every tick). Higher = lighter CPU, less
# frequent "who's near me" updates. Perception is 0-LLM regardless.
PERCEPTION_EVERY_N_TICKS = max(1, _env_int("SIM_PERCEPTION_EVERY_N_TICKS", 3))

# Simulation date the run starts on (YYYY-MM-DD). Used to look up / store that
# day's plans and memory. Change this to run/limit a specific day.
SIM_START_DATE = _env_str("SIM_START_DATE", "2026-07-03")

# Time of day the run starts at (HH:MM). 00:00 = midnight (agents asleep at their
# hostels); set e.g. 08:00 to begin during the day so movement is visible at once.
SIM_START_TIME = _env_str("SIM_START_TIME", "00:00")

# ---------------------------------------------------------------------------
# Simulation clock
#   SIM_MINUTES_PER_TICK        = sim-minutes advanced per engine tick.
#   REAL_SECONDS_PER_SIM_MINUTE = wall-clock seconds per 1 sim-minute.
#   REAL_SECONDS_PER_TICK       = derived (do NOT set manually).
#
# Default is 4.0 real-sec/sim-min => a full sim-day (1440 min) takes ~96 real
# minutes. Slowing the clock does not reduce total LLM calls per sim-day, but
# it spreads them over 4x more wall-clock time, easing the per-key 5-RPM free
# tier limit and reducing 429 cooldowns.
# ---------------------------------------------------------------------------
SIM_MINUTES_PER_TICK = _env_int("SIM_MINUTES_PER_TICK", 1)
REAL_SECONDS_PER_SIM_MINUTE = _env_float("SIM_REAL_SECONDS_PER_SIM_MINUTE", 4.0)
REAL_SECONDS_PER_TICK: float = SIM_MINUTES_PER_TICK * REAL_SECONDS_PER_SIM_MINUTE

# Global speed multiplier (2.0 = twice as fast, 0.5 = half). Applied on top of
# the clock above by the engine's run loop.
TICK_SPEED = _env_float("SIM_TICK_SPEED", 1.0)

# ---------------------------------------------------------------------------
# Cognition / LLM budget toggles
# ---------------------------------------------------------------------------

# Reflex (react) LLM escalation. LOCKED OFF for the demo — the brain's reflex
# path stays 0-LLM (heuristic "keep going"). Flip to true (SIM_REFLEX_LLM=true)
# to enable the whitelisted reflex escalations in the future.
REFLEX_LLM_ENABLED = _env_bool("SIM_REFLEX_LLM", False)

# At most one reflex LLM escalation per agent per this many sim-minutes.
REFLEX_RATE_LIMIT_MINUTES = _env_int("SIM_REFLEX_RATE_LIMIT_MINUTES", 30)

# Long-term memory is Qdrant-only. Short-term JSON is active-day operational
# state, never a long-term keyword-search fallback.
MEMORY_BACKEND = "vector"

# Cloud Qdrant long-term-memory settings. Credentials are deliberately read
# only from the environment. When unavailable, recall is empty and archival
# keeps its short-term source rather than creating a local long-term archive.
SEMANTIC_MEMORY_ENABLED = _env_bool("SIM_SEMANTIC_MEMORY_ENABLED", True)
QDRANT_URL = _env_str("QDRANT_URL", "")
QDRANT_API_KEY = _env_str("QDRANT_API_KEY", "")
MEMORY_EMBEDDING_MODEL = _env_str("SIM_MEMORY_EMBEDDING_MODEL", "gemini-embedding-001")
MEMORY_VECTOR_DIMENSIONS = _env_int("SIM_MEMORY_VECTOR_DIMENSIONS", 768)
MEMORY_COLLECTION_VERSION = _env_str("SIM_MEMORY_COLLECTION_VERSION", "v1")
MEMORY_RETRIEVAL_CANDIDATE_MULTIPLIER = max(1, _env_int("SIM_MEMORY_RETRIEVAL_CANDIDATE_MULTIPLIER", 4))
MEMORY_RECENCY_DECAY_DAYS = max(1.0, _env_float("SIM_MEMORY_RECENCY_DECAY_DAYS", 30.0))
# Application-side quota guard for Cloud Qdrant. Qdrant Cloud reports actual
# plan usage in its dashboard; this estimate keeps the index below the chosen
# local budget before that dashboard limit is approached.
MEMORY_MAX_STORAGE_GB = max(0.1, _env_float("SIM_MEMORY_MAX_STORAGE_GB", 4.0))
MEMORY_STORAGE_PRUNE_THRESHOLD = min(1.0, max(0.1, _env_float("SIM_MEMORY_STORAGE_PRUNE_THRESHOLD", 0.90)))
MEMORY_STORAGE_PRUNE_TARGET = min(MEMORY_STORAGE_PRUNE_THRESHOLD, max(0.05, _env_float("SIM_MEMORY_STORAGE_PRUNE_TARGET", 0.85)))

# Cap on full day-plan regenerations triggered mid-day per agent (budget guard).
MAX_REPLANS_PER_AGENT_PER_DAY = _env_int("SIM_MAX_REPLANS_PER_AGENT_PER_DAY", 3)

# Budget governor: soft ceiling on LLM calls per real hour across the whole sim.
# 0 = no ceiling. When exceeded, cognition degrades gracefully (skip reflex,
# defer replans) — the sim keeps running on the 0-LLM executor path.
LLM_HOURLY_CEILING = _env_int("SIM_LLM_HOURLY_CEILING", 0)

# Energy and emotion thresholds for LLM calls
DECIDE_MIN_ENERGY = _env_float("SIM_DECIDE_MIN_ENERGY", 0.1)
DECIDE_MIN_EMOTION = _env_float("SIM_DECIDE_MIN_EMOTION", 0.1)
# An observation may change every tick while agents travel together.  Limit
# each agent's advisory LLM reflex to avoid turning normal movement into an
# unbounded API stream; their action state machines still run every tick.
DECIDE_COOLDOWN_TICKS = _env_int("SIM_DECIDE_COOLDOWN_TICKS", 30)
CONVERSATION_MIN_ENERGY = _env_float("SIM_CONVERSATION_MIN_ENERGY", 0.05)
CONVERSATION_MIN_EMOTION = _env_float("SIM_CONVERSATION_MIN_EMOTION", 0.05)

# LLM Configuration
TEMPERATURE = 0.7           # Creativity the model is allowed
# Per-key RPM limit. Gemini Flash Lite's available free-tier limit is shown
# per project in AI Studio; use 15 by default and let the local .env lower it
# for projects with a smaller allocation.
API_RPM_LIMIT = max(1, _env_int("SIM_API_RPM_LIMIT", 15))
# Every individual Gemini HTTP request has a short deadline, and the complete
# fallback cascade has its own cap.  These prevent one unavailable model from
# holding a planner/conversation worker indefinitely.
# Gemini rejects manually configured deadlines below 10 seconds.  Keep the
# outer deadline long enough for a bounded primary/key fallback sequence.
LLM_REQUEST_TIMEOUT_MS = max(10_000, _env_int("SIM_LLM_REQUEST_TIMEOUT_MS", 10_000))
LLM_CALL_DEADLINE_SECONDS = max(10.0, _env_float("SIM_LLM_CALL_DEADLINE_SECONDS", 30.0))
LLM_TRANSIENT_KEY_COOLDOWN_SECONDS = max(
    1.0, _env_float("SIM_LLM_TRANSIENT_KEY_COOLDOWN_SECONDS", 15.0)
)
LLM_KEY_ACQUIRE_WAIT_SECONDS = max(
    0.0, _env_float("SIM_LLM_KEY_ACQUIRE_WAIT_SECONDS", 2.0)
)

# Model fallback chain. Order = preference order.
# All calls use the single "default" tier: primary gemini-3.1-flash-lite, with fallbacks.
# Multiple API keys provide rate-limit resilience (see API_KEYS below).
MODEL_TIERS: dict[str, list[str]] = {
    "default": ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3-flash-preview"],
}

# Support multiple API keys (comma-separated in env var).
# Rotates through them when rate limits are hit on the current key.
API_KEYS: list[str] = [
    k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()
]
if not API_KEYS:
    # Gather GEMINI_API_KEY (no suffix) + GEMINI_API_KEY_1 through _100
    # Only includes keys that are actually set (non-empty).
    numbered_keys = []
    for index in range(1, 101):
        key = os.environ.get(f"GEMINI_API_KEY_{index}", "")
        if key:
            numbered_keys.append(key)
    API_KEYS = [key for key in [os.environ.get("GEMINI_API_KEY", ""), *numbered_keys] if key]

# Drop obvious placeholder values so the client doesn't waste attempts (and
# incur cooldowns) on non-functional keys.
_PLACEHOLDER_MARKERS = ("your_google_api_key", "your_api_key", "changeme", "xxxx")
API_KEYS = [
    k for k in API_KEYS
    if not any(marker in k.lower() for marker in _PLACEHOLDER_MARKERS)
]


# Conversation cap (per agent per day)
MAX_CONVERSATIONS_PER_AGENT = _env_int("SIM_MAX_CONVERSATIONS_PER_AGENT", 5)

# At the calendar boundary, allow active conversation jobs this many real
# seconds to finish and persist before the engine cancels the remainder.
DAY_HANDOFF_CONVERSATION_TIMEOUT_SECONDS = _env_float(
    "SIM_DAY_HANDOFF_CONVERSATION_TIMEOUT_SECONDS", 45.0,
)

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


# ---------------------------------------------------------------------------
# CLI + runtime override surface
# ---------------------------------------------------------------------------
# Every toggle above has a default (hard-coded) that can be overridden by an
# environment variable (.env), which can in turn be overridden by a CLI flag.
# Precedence: CLI flag > .env > built-in default.


# Maps CLI/override keyword -> config global name it sets.
_OVERRIDE_MAP = {
    "tick_speed": "TICK_SPEED",
    "real_seconds_per_sim_minute": "REAL_SECONDS_PER_SIM_MINUTE",
    "sim_minutes_per_tick": "SIM_MINUTES_PER_TICK",
    "perception_radius_px": "PERCEPTION_RADIUS_PX",
    "perception_enabled": "PERCEPTION_ENABLED",
    "max_conversations_per_agent": "MAX_CONVERSATIONS_PER_AGENT",
    "max_replans_per_agent_per_day": "MAX_REPLANS_PER_AGENT_PER_DAY",
    "llm_hourly_ceiling": "LLM_HOURLY_CEILING",
    "decide_min_energy": "DECIDE_MIN_ENERGY",
    "decide_min_emotion": "DECIDE_MIN_EMOTION",
    "decide_cooldown_ticks": "DECIDE_COOLDOWN_TICKS",
    "conversation_min_energy": "CONVERSATION_MIN_ENERGY",
    "conversation_min_emotion": "CONVERSATION_MIN_EMOTION",
    "day_handoff_conversation_timeout_seconds": "DAY_HANDOFF_CONVERSATION_TIMEOUT_SECONDS",
}


def apply_overrides(**overrides) -> None:
    """Apply runtime overrides (typically parsed CLI args) onto the module
    globals. Only non-None values are applied. Recomputes derived constants."""
    global REAL_SECONDS_PER_TICK
    for key, value in overrides.items():
        if value is None:
            continue
        target = _OVERRIDE_MAP.get(key)
        if target is None:
            continue
        globals()[target] = value
    # Recompute derived clock constant.
    REAL_SECONDS_PER_TICK = SIM_MINUTES_PER_TICK * REAL_SECONDS_PER_SIM_MINUTE


def add_cli_arguments(parser) -> None:
    """Register the toggle flags on an argparse parser. Defaults are None so
    that an unspecified flag leaves the .env/built-in value untouched."""
    g = parser.add_argument_group("simulation toggles (override .env)")
    g.add_argument("--tick-speed", dest="tick_speed", type=float, default=None,
                   help="Speed multiplier (2.0 = 2x faster). Default 1.0")
    g.add_argument("--real-seconds-per-sim-minute", dest="real_seconds_per_sim_minute",
                   type=float, default=None,
                   help="Wall-clock seconds per sim-minute (7.5 => 180 real-min/day)")
    g.add_argument("--sim-minutes-per-tick", dest="sim_minutes_per_tick", type=int,
                   default=None, help="Sim-minutes advanced per tick (default 1)")
    g.add_argument("--perception-radius-px", dest="perception_radius_px", type=int,
                   default=None, help="Proximity circle radius in pixels (default 50)")
    g.add_argument("--perception", dest="perception_enabled", type=_str2bool, default=None,
                   help="Run the perception step each tick (default: on)")
    g.add_argument("--max-conversations", dest="max_conversations_per_agent", type=int,
                   default=None, help="Max conversations per agent per day (default 5)")
    g.add_argument("--max-replans", dest="max_replans_per_agent_per_day", type=int,
                   default=None, help="Max mid-day full replans per agent (default 3)")
    g.add_argument("--llm-hourly-ceiling", dest="llm_hourly_ceiling", type=int,
                   default=None, help="Soft LLM calls/real-hour ceiling (0 = none)")
    g.add_argument("--decide-min-energy", dest="decide_min_energy", type=float,
                   default=None, help="Min energy for LLM decide (default 0.1)")
    g.add_argument("--decide-min-emotion", dest="decide_min_emotion", type=float,
                   default=None, help="Min emotion for LLM decide (default 0.1)")
    g.add_argument("--conv-min-energy", dest="conversation_min_energy", type=float,
                   default=None, help="Min energy for conversation (default 0.05)")
    g.add_argument("--conv-min-emotion", dest="conversation_min_emotion", type=float,
                   default=None, help="Min emotion for conversation (default 0.05)")
    g.add_argument("--day-handoff-conversation-timeout", dest="day_handoff_conversation_timeout_seconds", type=float,
                   default=None, help="Real seconds to wait for conversations at day rollover (default 45)")


def _str2bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on", "y")


def overrides_from_args(args) -> dict:
    """Extract the override kwargs from a parsed argparse namespace."""
    return {key: getattr(args, key, None) for key in _OVERRIDE_MAP}


def describe_settings() -> str:
    """Human-readable one-block summary of all active simulation toggles."""
    # sim-day = 1440 sim-min; real seconds = 1440 * REAL_SECONDS_PER_SIM_MINUTE / TICK_SPEED
    day_real_min = (1440 * REAL_SECONDS_PER_SIM_MINUTE / max(TICK_SPEED, 1e-9)) / 60.0
    return (
        "Valhalla simulation settings\n"
        f"  API keys loaded              : {len(API_KEYS)}\n"
        f"  API RPM limit per key        : {API_RPM_LIMIT}\n"
        f"  Model tiers                  : default={MODEL_TIERS['default']}\n"
        f"  Memory backend               : {MEMORY_BACKEND}\n"
        f"  Semantic memory              : {'ON' if SEMANTIC_MEMORY_ENABLED else 'OFF'}\n"
        f"  Perception                   : {'ON' if PERCEPTION_ENABLED else 'OFF'} (radius {PERCEPTION_RADIUS_PX}px)\n"
        f"  Sim clock                    : {SIM_MINUTES_PER_TICK} sim-min/tick, "
        f"{REAL_SECONDS_PER_SIM_MINUTE}s/sim-min, speed x{TICK_SPEED}\n"
        f"  => real sec/tick             : {REAL_SECONDS_PER_TICK / max(TICK_SPEED,1e-9):.2f}s\n"
        f"  => real min per sim-day      : ~{day_real_min:.0f} min\n"
        f"  Max conversations/agent/day  : {MAX_CONVERSATIONS_PER_AGENT}\n"
        f"  Max replans/agent/day        : {MAX_REPLANS_PER_AGENT_PER_DAY}\n"
        f"  LLM hourly ceiling           : {LLM_HOURLY_CEILING or 'none'}\n"
        f"  Decide min energy/emotion    : {DECIDE_MIN_ENERGY} / {DECIDE_MIN_EMOTION}\n"
        f"  Conv min energy/emotion      : {CONVERSATION_MIN_ENERGY} / {CONVERSATION_MIN_EMOTION}\n"
        f"  Day-handoff conversation wait: {DAY_HANDOFF_CONVERSATION_TIMEOUT_SECONDS:.0f}s\n"
    )


if __name__ == '__main__':
    print(f"Running this project from : {BASE_DIR}")
    print(describe_settings())
