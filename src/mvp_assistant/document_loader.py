import re
from pathlib import Path

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _strip_html(html: str) -> str:
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def _read_pdf(path: Path) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""


def load_knowledge_base_corpus(kb_dir: Path) -> str:
    parts: list[str] = []
    for p in sorted(kb_dir.rglob("*")):
        if not p.is_file():
            continue

        suffix = p.suffix.lower()
        if suffix == ".pdf":
            text = _read_pdf(p)
        elif suffix in {".html", ".htm"}:
            text = _strip_html(_read_text_file(p))
        elif suffix in {".csv", ".txt", ".md"}:
            text = _read_text_file(p)
        else:
            continue

        if text:
            parts.append(f"\nFILE: {p.name}\n{text}\n")

    return "\n".join(parts)
