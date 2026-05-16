import re

from .text_utils import normalize_text


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" -:\t")


def _extract_restaurant(text: str, fallback: str) -> str:
    match = re.search(r"^(.{2,80})$", text[:500], flags=re.MULTILINE)
    if match:
        return _clean_line(match.group(1)).strip("\"“”")
    return fallback


def _extract_planet(text: str, planet_names: list[str] | None = None) -> str | None:
    header = text[:3500]
    normalized_header = normalize_text(header)
    for planet in planet_names or []:
        planet_norm = normalize_text(planet)
        if re.search(rf"\b{re.escape(planet_norm)}\b", normalized_header):
            return planet
    return None


def _extract_licenses(text: str) -> dict[str, int]:
    del text
    return {}


def _find_positions(text: str, dish_names: list[str]) -> list[tuple[int, str]]:
    positions: list[tuple[int, str]] = []
    for name in dish_names:
        parts = [re.escape(part) for part in re.split(r"\s+", name.strip()) if part]
        pattern = r"\s+".join(parts).replace("'", r"['’]")
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            positions.append((match.start(), name))
    return sorted(positions)


def _slice_segment(text: str, start: int, end: int) -> str:
    return text[start:end].strip()
