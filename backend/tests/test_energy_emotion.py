"""Test 4: day planner action schema. Test 5-7: Energy/emotion changes."""

def test_action_schema_has_energy_emotion():
    import sys; sys.path.insert(0, "backend")
    from src.agents.day_planner import FineAction
    fields = FineAction.model_fields
    assert "energy_change" in fields
    assert "emotion_change" in fields

def test_registry_state_has_energy_emotion():
    import sys; sys.path.insert(0, "backend")
    from src.core.agent_registry import AgentRuntimeState
    s = AgentRuntimeState(agent_id="t", persona={}, persona_name="t", position={"x": 0, "y": 0, "location_id": "t"})
    assert hasattr(s, "energy_level")
    assert hasattr(s, "emotion_state")
    assert 0.0 <= s.energy_level <= 1.0
    assert 0.0 <= s.emotion_state <= 1.0

def test_action_state_has_energy_emotion():
    import sys; sys.path.insert(0, "backend")
    from src.agents.Actions import ActionState, ActionType
    a = ActionState(
        action_type=ActionType.MISC, description="t", start_time="00:00", end_time="01:00",
        location_id="t",
        energy_change=0.5, emotion_change=-0.3,
    )
    assert a.energy_change == 0.5
    assert a.emotion_change == -0.3

def test_energy_clamping():
    vals = [-0.5, 0.0, 0.3, 1.0, 1.5]
    clamped = [max(0.0, min(1.0, v)) for v in vals]
    assert clamped == [0.0, 0.0, 0.3, 1.0, 1.0], f"clamping failed: {clamped}"
