import re
import unicodedata


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.lower().replace("’", "'")
    value = "".join(ch for ch in unicodedata.normalize("NFD", value) if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", value).strip()


def tokenize(value: str) -> list[str]:
    raw = re.findall(r"[a-z0-9\+']+", normalize_text(value))
    return [t for t in raw if len(t) > 1]
