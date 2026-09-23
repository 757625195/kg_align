#!/usr/bin/env python3
"""Build a portable archive of Codex sessions associated with this project."""

from __future__ import annotations

import json
import re
import tarfile
from datetime import datetime
from pathlib import Path


HOME = Path.home()
CODEX_DIR = HOME / ".codex"
PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "migration" / "kg_align_codex_sessions.tar.gz"
SESSION_ID = re.compile(r"([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})")


def contains_project(path: Path) -> bool:
    needle = str(PROJECT).encode()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if needle in chunk:
                return True
    return False


def main() -> None:
    candidates: list[Path] = []
    for root_name in ("sessions", "archived_sessions"):
        root = CODEX_DIR / root_name
        if root.exists():
            candidates.extend(root.rglob("*.jsonl"))

    selected = sorted(path for path in candidates if contains_project(path))
    ids = {
        match.group(1)
        for path in selected
        if (match := SESSION_ID.search(path.name)) is not None
    }

    index_entries: list[dict] = []
    index_path = CODEX_DIR / "session_index.jsonl"
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("id") in ids:
                index_entries.append(entry)

    total_bytes = sum(path.stat().st_size for path in selected)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_project": str(PROJECT),
        "session_count": len(selected),
        "uncompressed_bytes": total_bytes,
        "session_ids": sorted(ids),
    }
    readme = f"""kg_align Codex session migration

This archive contains {len(selected)} local Codex session files whose contents
reference:
  {PROJECT}

Restore on the new computer (with Codex fully closed):
1. Extract this archive to a temporary directory.
2. Copy the extracted .codex/sessions and .codex/archived_sessions contents into
   the corresponding directories under ~/.codex, preserving subdirectories.
3. Merge filtered_session_index.jsonl into ~/.codex/session_index.jsonl.
4. Start Codex. A session can also be opened with: codex resume <session-id>

The archive intentionally excludes auth.json, credentials, caches, plugins,
unrelated sessions, and project files. Sign in normally on the new computer.
Keep the project at the same absolute path when practical; otherwise tell Codex
the new project path after resuming a session.
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUTPUT, "w:gz", compresslevel=6) as archive:
        for path in selected:
            archive.add(path, arcname=str(Path(".codex") / path.relative_to(CODEX_DIR)))

        payloads = {
            "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            "filtered_session_index.jsonl": "".join(
                json.dumps(entry, ensure_ascii=False) + "\n" for entry in index_entries
            ),
            "README.txt": readme,
        }
        for name, content in payloads.items():
            encoded = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(encoded)
            info.mtime = int(datetime.now().timestamp())
            import io

            archive.addfile(info, io.BytesIO(encoded))

    print(f"sessions={len(selected)}")
    print(f"uncompressed_bytes={total_bytes}")
    print(f"archive={OUTPUT}")
    print(f"archive_bytes={OUTPUT.stat().st_size}")


if __name__ == "__main__":
    main()
