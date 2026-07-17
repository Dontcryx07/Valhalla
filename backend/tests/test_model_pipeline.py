"""Test 1: Verify model priority and fallback. Test 2: Verify no Gemma references."""

def test_no_gemma_in_config():
    import sys; sys.path.insert(0, "backend/src")
    from config import MODEL_TIERS
    for tier_name, models in MODEL_TIERS.items():
        for m in models:
            assert "gemma" not in m.lower(), f"Gemma model found in {tier_name}: {m}"

def test_model_priority():
    import sys; sys.path.insert(0, "backend/src")
    from config import MODEL_TIERS
    default = MODEL_TIERS["default"]
    assert default[0] == "gemini-3.1-flash-lite", f"primary should be gemini-3.1-flash-lite, got {default[0]}"
    assert "gemini-2.5-flash" in default, "2.5 flash must be in fallback chain"
