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


def test_location_resolver_rejects_missing_location_without_crashing():
    from src.agents.Actions import LocationResolver
    assert LocationResolver().resolve(None) is None

def test_energy_clamping():
    vals = [-0.5, 0.0, 0.3, 1.0, 1.5]
    clamped = [max(0.0, min(1.0, v)) for v in vals]
    assert clamped == [0.0, 0.0, 0.3, 1.0, 1.0], f"clamping failed: {clamped}"


def test_wellbeing_is_activity_driven_and_snapshot_is_direct():
    from types import SimpleNamespace
    from src.core.agent_registry import AgentRuntimeState
    from src.core.world_engine import WorldEngine
    from src.core.world_state import Position

    engine = WorldEngine()
    def state(agent_id):
        return AgentRuntimeState(
            agent_id=agent_id, persona={}, persona_name=agent_id,
            position=Position(x=1, y=2, location_id="hostel"),
            energy_level=0.75, emotion_state=0.50, emotion_baseline=0.50,
        )
    def action(description, start="09:00", end="10:00"):
        return SimpleNamespace(
            description=description, start_time=start, end_time=end,
            energy_change=0.0, emotion_change=0.0, action_type="misc",
        )

    academic = engine._action_wellbeing_deltas(state("academic"), action("Attend a CS lecture"))
    sleep = engine._action_wellbeing_deltas(state("sleep"), action("Sleep", "00:00", "07:00"))
    sport = engine._action_wellbeing_deltas(state("sport"), action("Play badminton with friends"))
    assert academic[0] < -0.05 and sleep[0] > 0.15
    assert sport[0] < -0.08 and sport[1] > 0.03
    # Variation is deterministic for rewind/checkpoint replay.
    assert academic == engine._action_wellbeing_deltas(state("academic"), action("Attend a CS lecture"))

    visible = state("visible")
    visible.energy_level, visible.emotion_state = 0.37, 0.64
    engine.registry.register(visible)
    payload = engine._frontend_snapshot(12, "00:12")["agents"]["visible"]
    assert payload["energy_level"] == 0.37
    assert payload["emotion_state"] == 0.64
