"""Read-only, non-LLM overnight monitor for a running Valhalla simulation.

It polls the same state endpoint used by the frontend and writes two files:
* ``*.jsonl``: every sampled snapshot and every detected finding (machine-readable)
* ``*.txt``: a concise, chronological report suitable for morning review

The monitor never imports simulation modules, calls an LLM, writes checkpoints,
or invokes any control endpoint. Stop it with Ctrl+C; it flushes a final summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fetch_json(url: str, timeout: float) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload, round((time.perf_counter() - started) * 1000, 1)


def _fetch_ui_health(url: str, timeout: float) -> tuple[int, int, float]:
    """Check that Odin is serving the frontend shell without a browser/LLM."""
    started = time.perf_counter()
    request = urllib.request.Request(url, headers={"Accept": "text/html"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        return response.status, len(body), round((time.perf_counter() - started) * 1000, 1)


def _agent_view(agent_id: str, agent: dict[str, Any]) -> dict[str, Any]:
    action = agent.get("current_action") or {}
    position = agent.get("position") or {}
    conversation = agent.get("conversation") or {}
    if not isinstance(conversation, dict):
        conversation = {"partner_name": str(conversation)}
    return {
        "agent_id": agent_id,
        "name": agent.get("name", agent_id),
        "location_id": position.get("location_id"),
        "position": {"x": position.get("x"), "y": position.get("y")},
        "activity": agent.get("activity"),
        "action_type": action.get("action_type"),
        "action_start": action.get("start_time"),
        "action_end": action.get("end_time"),
        "action_location_id": action.get("location_id"),
        "paused": bool(agent.get("paused")),
        "in_conversation": bool(agent.get("in_conversation")),
        "conversation_with": conversation.get("with") or conversation.get("partner_name"),
        "energy": agent.get("energy_level"),
        "emotion": agent.get("emotion_state"),
    }


class SidecarMonitor:
    def __init__(self, url: str, interval: float, timeout: float, output_dir: Path,
                 max_samples: int | None = None) -> None:
        self.base_url = url.rstrip("/")
        self.url = self.base_url + "/api/sim/state"
        self.interval = interval
        self.timeout = timeout
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = output_dir / f"sidecar_{stamp}.jsonl"
        self.report_path = output_dir / f"sidecar_{stamp}.txt"
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8", buffering=1)
        self._report = self.report_path.open("a", encoding="utf-8", buffering=1)
        self.previous_tick: int | None = None
        self.previous_agents: dict[str, dict[str, Any]] = {}
        self.paused_since: dict[str, int] = {}
        self.failures = 0
        self.findings: Counter[str] = Counter()
        self.samples = 0
        self.max_samples = max_samples

    def write(self, kind: str, data: dict[str, Any]) -> None:
        record = {"observed_at": _utc_now(), "kind": kind, **data}
        self._jsonl.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def note(self, level: str, message: str, **data: Any) -> None:
        self.findings[level] += 1
        self.write("finding", {"level": level, "message": message, **data})
        self._report.write(f"[{_utc_now()}] {level.upper()}: {message}\n")

    def observe(self, snapshot: dict[str, Any], latency_ms: float) -> None:
        self.samples += 1
        tick = snapshot.get("tick")
        agents = snapshot.get("agents") or {}
        health = snapshot.get("health") or {}
        running = (snapshot.get("simulation") or {}).get("running")
        agent_views = {agent_id: _agent_view(agent_id, agent) for agent_id, agent in agents.items()}
        self.write("snapshot", {
            "tick": tick, "time": snapshot.get("time"), "day": snapshot.get("day"),
            "running": running, "latency_ms": latency_ms, "health": health,
            "recent_conversations": snapshot.get("recent_conversations", []),
            "events": snapshot.get("events", {}), "agents": agent_views,
        })

        # UI health is represented by the exact payload the frontend consumes.
        if not isinstance(agents, dict) or not isinstance(health, dict):
            self.note("error", "frontend state payload is malformed", tick=tick)
            return
        if running is not True:
            self.note("warning", "simulation reports not running", tick=tick)
        if health.get("healthy") is False or health.get("anomalies"):
            self.note("error", "server health monitor reported anomalies", tick=tick,
                      anomalies=health.get("anomalies", []))
        if self.previous_tick is not None and isinstance(tick, int) and running and tick <= self.previous_tick:
            self.note("warning", "simulation tick did not advance", previous_tick=self.previous_tick, tick=tick)

        for agent_id, view in agent_views.items():
            if not view["location_id"] or not view["activity"] or not view["action_type"]:
                self.note("error", f"{view['name']} has incomplete rendered state", tick=tick, agent=view)
            if view["action_type"] != "move" and view["action_location_id"] and view["location_id"] != view["action_location_id"]:
                self.note("warning", f"{view['name']} location/action mismatch", tick=tick, agent=view)
            if view["paused"] or view["in_conversation"]:
                self.paused_since.setdefault(agent_id, tick if isinstance(tick, int) else -1)
                started = self.paused_since[agent_id]
                if isinstance(tick, int) and started >= 0 and tick - started >= 30:
                    self.note("warning", f"{view['name']} paused or conversing for 30+ ticks", tick=tick, agent=view)
            else:
                self.paused_since.pop(agent_id, None)

            before = self.previous_agents.get(agent_id)
            if before and any(before.get(field) != view.get(field) for field in ("location_id", "activity", "in_conversation", "paused")):
                self.write("agent_transition", {"tick": tick, "time": snapshot.get("time"),
                                                "before": before, "after": view})

        self.previous_tick = tick if isinstance(tick, int) else self.previous_tick
        self.previous_agents = agent_views
        try:
            status, bytes_served, ui_latency = _fetch_ui_health(self.base_url + "/", self.timeout)
            self.write("ui_health", {"tick": tick, "status": status, "bytes": bytes_served,
                                     "latency_ms": ui_latency, "healthy": status == 200 and bytes_served > 0})
            if status != 200 or bytes_served <= 0:
                self.note("error", "frontend shell is not healthy", tick=tick, status=status, bytes=bytes_served)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.note("error", f"frontend shell unavailable: {exc}", tick=tick)
        self._report.write(
            f"[{_utc_now()}] tick={tick} day={snapshot.get('day')} time={snapshot.get('time')} "
            f"agents={len(agent_views)} healthy={health.get('healthy')} latency={latency_ms}ms "
            f"moving={health.get('moving')} paused={health.get('paused')} conversations={health.get('conversations')}\n"
        )

    def run(self) -> None:
        self._report.write(f"Sidecar started: endpoint={self.url}, interval={self.interval}s\n")
        try:
            while True:
                try:
                    snapshot, latency_ms = _fetch_json(self.url, self.timeout)
                    self.observe(snapshot, latency_ms)
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                    self.failures += 1
                    self.note("error", f"state endpoint unavailable: {exc}")
                if self.max_samples is not None and self.samples >= self.max_samples:
                    self._report.write("Sidecar reached requested sample limit.\n")
                    break
                time.sleep(self.interval)
        except KeyboardInterrupt:
            self._report.write("Sidecar stopped by user.\n")
        finally:
            self.write("summary", {"samples": self.samples, "endpoint_failures": self.failures,
                                   "findings": dict(self.findings)})
            self._report.write(
                f"Summary: samples={self.samples}, endpoint_failures={self.failures}, "
                f"findings={dict(self.findings)}\n"
            )
            self._jsonl.close()
            self._report.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Valhalla simulation sidecar monitor")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Odin base URL")
    parser.add_argument("--interval", type=float, default=5.0, help="poll interval in seconds")
    parser.add_argument("--timeout", type=float, default=4.0, help="HTTP timeout in seconds")
    parser.add_argument("--output-dir", type=Path, default=Path("backend/output/sidecar_monitor"))
    parser.add_argument("--max-samples", type=int, help="stop after this many successful samples (test/helper mode)")
    args = parser.parse_args()
    if args.interval <= 0 or args.timeout <= 0:
        parser.error("--interval and --timeout must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    monitor = SidecarMonitor(args.url, args.interval, args.timeout, args.output_dir, args.max_samples)
    print(f"Writing monitor output to {monitor.jsonl_path} and {monitor.report_path}")
    monitor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
