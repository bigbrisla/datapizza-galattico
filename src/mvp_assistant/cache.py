import hashlib
import json
from pathlib import Path
from typing import Any


class JsonCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def key_for(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self.root / f"{key}.json"
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
