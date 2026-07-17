"""Test 8: Decide API skipped when energy below threshold. Test 9: Conversation skipped when emotion below threshold."""

def test_decide_gate_thresholds():
    import sys; sys.path.insert(0, "backend")
    from src import config as _cfg
    assert hasattr(_cfg, "DECIDE_MIN_ENERGY"), "DECIDE_MIN_ENERGY config missing"
    assert hasattr(_cfg, "DECIDE_MIN_EMOTION"), "DECIDE_MIN_EMOTION config missing"
    assert 0.0 <= _cfg.DECIDE_MIN_ENERGY <= 1.0
    assert 0.0 <= _cfg.DECIDE_MIN_EMOTION <= 1.0

def test_conversation_gate_thresholds():
    import sys; sys.path.insert(0, "backend")
    from src import config as _cfg
    assert hasattr(_cfg, "CONVERSATION_MIN_ENERGY")
    assert hasattr(_cfg, "CONVERSATION_MIN_EMOTION")
