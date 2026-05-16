from .models import ParsedMenu, SemanticChunk
from .structured_parser import _find_positions, _slice_segment


class SemanticMenuChunker:
    """Semantic chunking step: one chunk per dish, enriched with menu metadata."""

    def chunk(self, menu: ParsedMenu, dish_mapping: dict[str, int], seen_ids: set[int]) -> list[SemanticChunk]:
        chunks: list[SemanticChunk] = []
        positions = _find_positions(menu.text, list(dish_mapping))

        for idx, (start, dish_name) in enumerate(positions):
            end = positions[idx + 1][0] if idx + 1 < len(positions) else min(len(menu.text), start + 2500)
            dish_id = dish_mapping[dish_name]
            if dish_id in seen_ids:
                continue
            seen_ids.add(dish_id)

            chunks.append(
                SemanticChunk(
                    dish_id=dish_id,
                    dish_name=dish_name,
                    source_document=menu.source_document,
                    restaurant=menu.restaurant,
                    planet=menu.planet,
                    licenses=menu.licenses,
                    text=_slice_segment(menu.text, start, end),
                )
            )

        return chunks
