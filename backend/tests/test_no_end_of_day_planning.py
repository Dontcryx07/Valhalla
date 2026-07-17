"""Test 3: Verify end-of-day planning no longer executes."""

def test_no_next_day_planning_in_check_last_action():
    import sys; sys.path.insert(0, "backend")
    import inspect
    from src.core.world_engine import WorldEngine
    source = inspect.getsource(WorldEngine._check_last_action_triggers)
    assert "next_day" not in source, "next_day mode should not appear in _check_last_action_triggers"
    assert "archiving day" in source, "docstring should mention archiving only"
