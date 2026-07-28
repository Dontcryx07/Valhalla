"""Operate Qdrant-only long-term memory.

``migrate-json`` imports legacy Long_term_db archives once.  Pass
``--delete-source`` only after checking the reported indexed count; it removes
the obsolete JSON archives after their idempotent Qdrant upsert succeeds.

``clear`` removes only Valhalla's durable Qdrant collections. It never touches
short-term runtime files, checkpoints, or the live simulation process.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.agents.vector_memory import VectorMemoryRetriever, archive_records
from src.config import DATA_DIR


def _legacy_archives() -> tuple[dict[str, list], list[Path]]:
    root = DATA_DIR / "Long_term_db"
    result: dict[str, list] = defaultdict(list)
    sources: list[Path] = []
    for path in root.glob("*/memory.json"):
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        agent = str(content.get("persona_name") or path.parent.name)
        for day in content.get("days", []):
            result[agent].extend(archive_records(agent, day))
        sources.append(path)
    return result, sources


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Qdrant-only long-term memory")
    parser.add_argument("operation", choices=("status", "migrate-json", "clear"))
    parser.add_argument("--agent", help="limit work to one persona name")
    parser.add_argument("--delete-source", action="store_true",
                        help="after a successful migrate-json, delete legacy memory.json files")
    parser.add_argument("--yes", action="store_true",
                        help="required for clear; confirms permanent deletion of Valhalla Qdrant memory")
    args = parser.parse_args()
    index = VectorMemoryRetriever()
    if not index.available:
        print("Qdrant memory is unavailable. Set SIM_SEMANTIC_MEMORY_ENABLED=true, QDRANT_URL, QDRANT_API_KEY, and Gemini keys.")
        return 2
    if args.operation == "clear":
        if args.agent:
            parser.error("clear always removes all Valhalla long-term memory; --agent is not supported")
        if not args.yes:
            parser.error("clear is permanent; rerun with --yes to remove all Valhalla long-term memory")
        result = index.clear_all_memory()
        deleted = result["deleted_collections"]
        failed = result["failed_collections"]
        print(f"Cleared {len(deleted)} Valhalla long-term-memory collection(s): {', '.join(deleted) or '(none found)'}")
        if failed:
            print(f"Could not clear {len(failed)} collection(s): {', '.join(failed)}")
            return 1
        return 0

    archives, sources = _legacy_archives()
    if args.agent:
        archives = {args.agent: archives.get(args.agent, [])}
        sources = [p for p in sources if p.parent.name == args.agent.lower().replace(" ", "_")]
    if args.operation == "status":
        agents = sorted(archives) if archives else ([args.agent] if args.agent else [])
        for agent in agents:
            print(f"{agent}: {index.collection_status(agent)}")
        return 0

    successful = True
    for agent, records in sorted(archives.items()):
        indexed = index.index_records(agent, records)
        expected = len({record.id for record in records})
        print(f"{agent}: indexed {indexed}/{expected} legacy memories")
        successful = successful and indexed == expected
    if args.delete_source:
        if not successful:
            print("Legacy JSON was retained because migration was incomplete.")
            return 1
        for path in sources:
            path.unlink()
            try:
                path.parent.rmdir()
            except OSError:
                pass
        print(f"Deleted {len(sources)} migrated legacy memory.json file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
