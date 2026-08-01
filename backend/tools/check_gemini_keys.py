"""Check every Gemini API key declared in the project .env file.

The script reads the same key layout used by the app, then validates each key
with a lightweight Gemini API call. It prints one line per key and exits with a
non-zero status if any key fails.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from google import genai
from google.genai import errors, types


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

PLACEHOLDER_MARKERS = ("your_google_api_key", "your_api_key", "changeme", "xxxx")


@dataclass(frozen=True)
class KeyEntry:
    label: str
    value: str


@dataclass(frozen=True)
class CheckResult:
    entry: KeyEntry
    ok: bool
    reason: str


def _parse_env_file(env_file: Path) -> dict[str, str]:
    if not env_file.exists():
        raise FileNotFoundError(f"Environment file not found: {env_file}")

    values: dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def _discover_keys(env_values: dict[str, str]) -> list[KeyEntry]:
    entries: list[KeyEntry] = []

    grouped = env_values.get("GEMINI_API_KEYS", "").strip()
    if grouped:
        for index, key in enumerate(grouped.split(","), start=1):
            cleaned = key.strip()
            if cleaned:
                entries.append(KeyEntry(label=f"GEMINI_API_KEYS[{index}]", value=cleaned))
        return [entry for entry in entries if not _looks_like_placeholder(entry.value)]

    base_key = env_values.get("GEMINI_API_KEY", "").strip()
    if base_key:
        entries.append(KeyEntry(label="GEMINI_API_KEY", value=base_key))

    for index in range(1, 101):
        key_name = f"GEMINI_API_KEY_{index}"
        key_value = env_values.get(key_name, "").strip()
        if key_value:
            entries.append(KeyEntry(label=key_name, value=key_value))

    return [entry for entry in entries if not _looks_like_placeholder(entry.value)]


def _status_code(exc: Exception) -> int | None:
    for attr_name in ("code", "status_code"):
        value = getattr(exc, attr_name, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    return None


def _response_message(exc: Exception) -> str | None:
    response_json = getattr(exc, "response_json", None)
    if isinstance(response_json, dict):
        error = response_json.get("error")
        if isinstance(error, dict):
            for field in ("message", "status", "detail"):
                value = error.get(field)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        for field in ("message", "status", "detail"):
            value = response_json.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()

    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", None)
    if isinstance(response_text, str) and response_text.strip():
        return response_text.strip()

    message = str(exc).strip()
    return message or None


def _friendly_reason(exc: Exception) -> str:
    status_code = _status_code(exc)
    message = _response_message(exc) or exc.__class__.__name__

    if isinstance(exc, (TimeoutError, genai.errors.UnknownApiResponseError)):
        return f"request timed out or the API returned an unreadable response: {message}"

    if isinstance(exc, (errors.ClientError, errors.APIError)):
        if status_code in {401, 403}:
            return f"authentication failed (HTTP {status_code}): {message}"
        if status_code == 429:
            return f"quota or rate limit reached (HTTP 429): {message}"
        if status_code is not None:
            return f"API returned HTTP {status_code}: {message}"

    if isinstance(exc, errors.ServerError):
        if status_code is not None:
            return f"Gemini server error (HTTP {status_code}): {message}"
        return f"Gemini server error: {message}"

    if status_code in {401, 403}:
        return f"authentication failed (HTTP {status_code}): {message}"
    if status_code == 429:
        return f"quota or rate limit reached (HTTP 429): {message}"
    if status_code is not None:
        return f"HTTP {status_code}: {message}"

    return message


def _check_key(entry: KeyEntry) -> CheckResult:
    try:
        client = genai.Client(api_key=entry.value)
        client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents="ping",
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1,
            ),
        )
        return CheckResult(entry=entry, ok=True, reason="valid")
    except Exception as exc:  # noqa: BLE001 - we want the exact provider failure reason
        return CheckResult(entry=entry, ok=False, reason=_friendly_reason(exc))


def _format_result(result: CheckResult) -> str:
    if result.ok:
        return f"[OK] {result.entry.label}"
    return f"[FAIL] {result.entry.label} -> {result.reason}"


def _summarize(results: Iterable[CheckResult]) -> tuple[int, int]:
    total = 0
    failures = 0
    for result in results:
        total += 1
        failures += int(not result.ok)
    return total, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate every Gemini API key declared in the project .env file",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Path to the .env file to inspect",
    )
    args = parser.parse_args()

    try:
        env_values = _parse_env_file(args.env_file)
    except FileNotFoundError as exc:
        print(str(exc))
        return 2

    entries = _discover_keys(env_values)
    if not entries:
        print(f"No Gemini API keys were found in {args.env_file}")
        return 2

    results = [_check_key(entry) for entry in entries]
    for result in results:
        print(_format_result(result))

    total, failures = _summarize(results)
    passed = total - failures
    print(f"\nSummary: {passed}/{total} keys valid")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())