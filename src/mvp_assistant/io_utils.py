import csv
import json
from pathlib import Path


def load_questions(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["domanda"] for row in reader]


def load_question_items(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{"domanda": row["domanda"], "difficoltà": row.get("difficoltà", "")} for row in reader]


def load_mapping(path: Path) -> dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_submission(path: Path, rows: list[tuple[int, list[int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "result"])
        for row_id, dish_ids in rows:
            writer.writerow([row_id, ",".join(str(x) for x in dish_ids)])
