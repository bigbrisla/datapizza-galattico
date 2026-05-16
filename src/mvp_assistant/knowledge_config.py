import csv
from pathlib import Path


class KnowledgeConfig:
    """Configuration discovered from the provided knowledge base, not hard-coded."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def planet_names(self) -> list[str]:
        distances_path = self.repo_root / "Dataset" / "knowledge_base" / "misc" / "Distanze.csv"
        if not distances_path.exists():
            return []

        with distances_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])

        return [value.strip() for value in header[1:] if value.strip()]
