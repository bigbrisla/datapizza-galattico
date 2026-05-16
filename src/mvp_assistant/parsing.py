from .models import ParsedMenu, SourceDocument
from .structured_parser import _extract_licenses, _extract_planet, _extract_restaurant


class MenuParser:
    """Parsing step: extract document-level metadata shared by all dish chunks."""

    def __init__(self, planet_names: list[str] | None = None) -> None:
        self.planet_names = planet_names or []

    def parse(self, document: SourceDocument) -> ParsedMenu:
        fallback_name = document.source_document.rsplit(".", 1)[0]
        return ParsedMenu(
            source_document=document.source_document,
            text=document.text,
            restaurant=_extract_restaurant(document.text, fallback_name),
            planet=_extract_planet(document.text, self.planet_names),
            licenses=_extract_licenses(document.text),
        )
