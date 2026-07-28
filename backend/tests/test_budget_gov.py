"""Test 10: Budget governor blocks decide and replan."""

def test_budget_governor_blocks():
    import sys; sys.path.insert(0, "backend")
    from src.core.budget import BudgetGovernor
    g = BudgetGovernor()
    for _ in range(10):
        g.record("test", "dummy")
    import src.config as _cfg
    original = _cfg.LLM_HOURLY_CEILING
    _cfg.LLM_HOURLY_CEILING = 5
    try:
        assert not g.can_afford("decide", cost=1), "Should deny when over ceiling"
        assert not g.can_afford("replan", cost=4), "Should deny replan when over ceiling"
    finally:
        _cfg.LLM_HOURLY_CEILING = original
