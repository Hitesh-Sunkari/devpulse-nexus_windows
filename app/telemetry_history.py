import json
import os
import threading
from collections import deque
from datetime import datetime
from pathlib import Path


HISTORY_FILE = Path(os.getenv(
    "TELEMETRY_HISTORY_PATH",
    Path(__file__).resolve().parent.parent / "telemetry_history.json",
))

MAX_RECORDS = 720

_history = deque(maxlen=MAX_RECORDS)
_lock = threading.Lock()


def _load_history():
    if not HISTORY_FILE.exists():
        return

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            records = json.load(file)

        if isinstance(records, list):
            with _lock:
                _history.extend(records[-MAX_RECORDS:])

    except (OSError, json.JSONDecodeError):
        pass


def _save_history():
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = HISTORY_FILE.with_suffix(".tmp")

        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(list(_history), file, indent=2)

        temporary_file.replace(HISTORY_FILE)

    except OSError:
        pass


_load_history()


def record_telemetry(digital_twin):
    record = {
        "timestamp": digital_twin.get(
            "timestamp",
            datetime.now().isoformat()
        ),
        "system": digital_twin.get("system", {}),
        "docker": digital_twin.get("docker", {})
    }

    with _lock:
        _history.append(record)
        _save_history()

    return record


def get_history(limit=120):
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 120

    limit = max(1, min(limit, MAX_RECORDS))

    with _lock:
        return list(_history)[-limit:]


def clear_history():
    with _lock:
        _history.clear()

        try:
            if HISTORY_FILE.exists():
                HISTORY_FILE.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "records": len(get_history()),
                "file": str(HISTORY_FILE)
            },
            indent=2
        )
    )
