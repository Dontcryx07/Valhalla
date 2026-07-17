"""
LLM budget governor -- one place that knows how much LLM spend has happened
recently and whether the simulation can afford more.

Why this exists
---------------
Free-tier Gemini keys are scarce (a handful of keys, a few requests/minute
each). Turning on perception + proximity conversations + a decision-making
brain could, if left ungated, burn the whole quota in minutes. Every
cognitive call site (day planner, conversation, reflex escalation) asks the
governor `can_afford()` before spending, and calls `record()` after. When the
soft ceiling is exceeded the governor says "no", and the caller degrades
gracefully -- the simulation keeps running on its 0-LLM executor path.

The governor is intentionally simple: a rolling one-real-hour window of call
timestamps, plus lifetime counters for observability (used by the budget
stress test and the on-screen/logged stats).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict

from src.core.log import get_logger
from src import config as _cfg

logger = get_logger(__name__)

_HOUR_SECONDS = 3600.0


class BudgetGovernor:
    """Thread-safe rolling-window LLM call accountant."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recent: Deque[float] = deque()      # monotonic timestamps, last hour
        self._total_calls = 0
        self._by_kind: Dict[str, int] = {}
        self._by_model: Dict[str, int] = {}
        self._denied = 0

    def _prune(self, now: float) -> None:
        cutoff = now - _HOUR_SECONDS
        while self._recent and self._recent[0] < cutoff:
            self._recent.popleft()

    def calls_last_hour(self) -> int:
        with self._lock:
            self._prune(time.monotonic())
            return len(self._recent)

    def can_afford(self, kind: str = "generic", cost: int = 1) -> bool:
        """True if a call of `kind` (costing `cost` LLM calls) is within the
        configured hourly ceiling. A ceiling of 0 means 'no limit'."""
        ceiling = _cfg.LLM_HOURLY_CEILING
        if not ceiling or ceiling <= 0:
            return True
        with self._lock:
            self._prune(time.monotonic())
            allowed = len(self._recent) + cost <= ceiling
        if not allowed:
            with self._lock:
                self._denied += 1
            logger.info(
                "[Budget] denied '%s' (cost=%d): %d/%d calls used this hour",
                kind, cost, self.calls_last_hour(), ceiling,
            )
        return allowed

    def record(self, kind: str = "generic", model: str = "unknown") -> None:
        """Record one successful LLM call."""
        now = time.monotonic()
        with self._lock:
            self._recent.append(now)
            self._prune(now)
            self._total_calls += 1
            self._by_kind[kind] = self._by_kind.get(kind, 0) + 1
            self._by_model[model] = self._by_model.get(model, 0) + 1

    def stats(self) -> Dict[str, object]:
        with self._lock:
            self._prune(time.monotonic())
            return {
                "total_calls": self._total_calls,
                "calls_last_hour": len(self._recent),
                "hourly_ceiling": _cfg.LLM_HOURLY_CEILING or None,
                "denied": self._denied,
                "by_kind": dict(self._by_kind),
                "by_model": dict(self._by_model),
            }

    def reset(self) -> None:
        with self._lock:
            self._recent.clear()
            self._total_calls = 0
            self._by_kind.clear()
            self._by_model.clear()
            self._denied = 0


# Process-wide singleton.
GOVERNOR = BudgetGovernor()


if __name__ == "__main__":
    from src.core.log import setup_logging
    setup_logging(run_id="budget_test", console=True)

    g = BudgetGovernor()
    assert g.can_afford("plan")
    for _ in range(5):
        g.record("plan", "gemini-3.1-flash-lite")
    assert g.calls_last_hour() == 5
    s = g.stats()
    assert s["total_calls"] == 5 and s["by_kind"]["plan"] == 5
    print("budget.py sanity check passed:", s)
