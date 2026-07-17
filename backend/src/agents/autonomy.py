"""
Autonomy — schema for the per-minute LLM decision to deviate from the plan.

The brain calls this once per agent per minute (when enabled). The LLM
sees the agent's persona, current plan, and nearby surroundings, then
decides whether to continue the plan or deviate temporarily.
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import Optional


class AutonomyDecision(BaseModel):
    """Structured output from the autonomy LLM call."""
    deviate: bool
    deviation_type: Optional[str] = None
    reason: Optional[str] = None
    duration_minutes: int = 0
