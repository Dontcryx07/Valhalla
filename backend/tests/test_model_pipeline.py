"""Verify model priority and absence of retired model families."""

from pathlib import Path
import sys
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from src.config import GEMINI_MODEL
from src.agents.day_planner import (
    CoarseBlock,
    CoarsePlanOutput,
    generate_coarse_plan,
    validate_coarse_window,
    validate_hourly_refinement,
    _force_accept,
    _within_source_windows,
    _local_academic_venue_check,
    _coarse_output_schema,
)

def test_no_gemma_in_config():
    assert "gemma" not in GEMINI_MODEL.lower()

def test_model_priority_and_retired_model_exclusion():
    assert GEMINI_MODEL == "gemini-3.1-flash-lite"
    assert GEMINI_MODEL != "gemini-2.5-flash", "retired 2.5 flash must not be attempted"


def test_remaining_plan_window_is_rejected_before_refinement():
    result = validate_coarse_window({
        "mode": "remaining",
        "current_time": "2026-07-03 20:06",
        "retry_count": 0,
        "coarse_plan": [{"activity": "Old schedule", "start": "00:00", "end": "24:00"}],
    })

    assert result["conflict_detected"] is True
    assert result["retry_count"] == 1
    assert "20:06" in result["conflict_reason"]


def test_remaining_prompt_contains_exact_window_constraint():
    state = {
        "persona": {"Name": "Test Student"},
        "mode": "remaining",
        "current_time": "2026-07-03 20:06",
        "current_location_id": "library",
        "relevant_memories": [],
        "yesterday_summary": None,
        "upcoming_events": [],
    }
    output = CoarsePlanOutput(blocks=[CoarseBlock(
        activity="Wind down", start="20:06", end="24:00", granularity="atomic"
    )], first_start_time="20:06")
    with patch("src.agents.day_planner.call_gemini", return_value=output) as call:
        generate_coarse_plan(state)

    prompt = call.call_args.args[1]
    assert "FIRST output block/action MUST start exactly at 20:06" in prompt
    assert "before 20:06" in prompt


def test_coarse_response_schema_binds_first_start_time_per_api_call():
    schema = _coarse_output_schema("20:06")
    json_schema = schema.model_json_schema()

    assert json_schema["properties"]["first_start_time"]["const"] == "20:06"
    valid = schema.model_validate({
        "first_start_time": "20:06",
        "blocks": [{
            "activity": "Wind down", "start": "20:06", "end": "24:00",
            "granularity": "atomic",
        }],
    })
    assert valid.blocks[0].start == "20:06"


def test_coarse_response_schema_rejects_first_block_mismatch():
    schema = _coarse_output_schema("20:06")
    try:
        schema.model_validate({
            "first_start_time": "20:06",
            "blocks": [{
                "activity": "Incorrect start", "start": "20:07", "end": "24:00",
                "granularity": "atomic",
            }],
        })
    except ValueError as exc:
        assert "first_start_time" in str(exc)
    else:
        raise AssertionError("schema accepted a coarse block with the wrong first start")


def test_hourly_refinement_cannot_recreate_an_atomic_block():
    state = {
        "mode": "next_day", "current_time": "2026-07-04 00:00", "retry_count": 0,
        "coarse_plan": [
            {"activity": "Sleep", "start": "00:00", "end": "07:00", "granularity": "atomic"},
            {"activity": "Morning routine", "start": "07:00", "end": "24:00", "granularity": "flexible"},
        ],
        "hourly_plan": [
            {"activity": "Sleep", "parent_activity": "Sleep", "start": "00:00", "end": "07:00", "granularity": "atomic"},
            {"activity": "Duplicate sleep", "parent_activity": "Sleep", "start": "00:00", "end": "07:00", "granularity": "flexible"},
            {"activity": "Morning routine", "parent_activity": "Morning routine", "start": "07:00", "end": "24:00", "granularity": "flexible"},
        ],
    }
    result = validate_hourly_refinement(state)
    assert result["conflict_detected"] is True
    assert result["retry_count"] == 1


def test_refinement_scope_filter_excludes_atomic_midnight_block():
    flexible_sources = [{"activity": "Morning routine", "start": "07:00", "end": "08:00"}]
    assert _within_source_windows(
        {"activity": "Shower", "start": "07:00", "end": "07:30"}, flexible_sources
    ) is True
    assert _within_source_windows(
        {"activity": "Duplicate sleep", "start": "00:00", "end": "07:00"}, flexible_sources
    ) is False


def test_invalid_full_day_plan_gets_valid_local_fallback():
    result = _force_accept({
        "mode": "next_day", "persona": {"Hostel": "hostel"},
        "current_location_id": "hostel", "places": [{"id": "hostel", "name": "Hostel", "type": "hostel"}],
        "conflict_reason": "invalid location", "fine_plan": [],
    })
    plan = result["day_plan"]
    assert [(item["start"], item["end"]) for item in plan] == [("00:00", "07:00"), ("07:00", "24:00")]
    assert {item["location_id"] for item in plan} == {"hostel"}


def test_academic_venue_policy_allows_preparation_before_lecture():
    issue = _local_academic_venue_check(
        [{"action": "Dressing and prepping for lectures", "location_id": "hostel"}],
        {"Branch": "Mechanical Engineering"},
    )
    assert issue is None


def test_academic_venue_policy_allows_library_note_review():
    issue = _local_academic_venue_check(
        [{
            "action": "Reviewing lecture notes and organizing study material",
            "location_id": "library",
        }],
        {"Branch": "Chemical Engineering"},
    )
    assert issue is None


def test_academic_venue_policy_allows_class_group_chat_at_hostel():
    issue = _local_academic_venue_check(
        [{
            "action": "Ignoring class group chats while lounging",
            "location_id": "Brahmaputra_Boys 1",
        }],
        {"Branch": "Artificial Intelligence and Data Engineering"},
    )
    assert issue is None
