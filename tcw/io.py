from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    resolved = ensure_parent(path)
    resolved.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return resolved


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_text(path: str | Path, content: str) -> Path:
    resolved = ensure_parent(path)
    resolved.write_text(content)
    return resolved
