"""Create rotating, portable backups of DevPulse persistent data."""

import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


SOURCE_DIRECTORIES = {
    "chroma": Path(os.getenv("CHROMA_BACKUP_SOURCE", "/data/chroma")),
    "telemetry": Path(os.getenv("TELEMETRY_BACKUP_SOURCE", "/data/telemetry")),
}
BACKUP_DIRECTORY = Path(os.getenv("BACKUP_DIRECTORY", "/backups"))
BACKUP_INTERVAL_SECONDS = int(os.getenv("BACKUP_INTERVAL_SECONDS", "86400"))
KEEP_BACKUPS = int(os.getenv("BACKUP_KEEP", "14"))


def create_backup():
    BACKUP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = BACKUP_DIRECTORY / f"devpulse-nexus-{timestamp}"
    staging.mkdir()

    for name, source in SOURCE_DIRECTORIES.items():
        if source.exists():
            shutil.copytree(source, staging / name, dirs_exist_ok=True)

    archive_base = BACKUP_DIRECTORY / staging.name
    archive = shutil.make_archive(str(archive_base), "zip", staging)
    shutil.rmtree(staging)

    archives = sorted(BACKUP_DIRECTORY.glob("devpulse-nexus-*.zip"))
    for old_archive in archives[:-KEEP_BACKUPS]:
        old_archive.unlink(missing_ok=True)

    print(f"Backup created: {archive}", flush=True)
    return archive


def run():
    while True:
        try:
            create_backup()
        except Exception as exc:
            print(f"Backup failed: {exc}", flush=True)
        time.sleep(BACKUP_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()
