"""Verify model priority and absence of retired model families."""

from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from src.config import MODEL_TIERS

def test_no_gemma_in_config():
    for tier_name, models in MODEL_TIERS.items():
        for m in models:
            assert "gemma" not in m.lower(), f"Gemma model found in {tier_name}: {m}"

def test_model_priority_and_retired_model_exclusion():
    default = MODEL_TIERS["default"]
    assert default[0] == "gemini-3.1-flash-lite", f"primary should be gemini-3.1-flash-lite, got {default[0]}"
    assert "gemini-2.5-flash" not in default, "retired 2.5 flash must not be attempted"
