from .models import DishDoc, QuerySpec
from .text_utils import normalize_text


class LexicalRetriever:
    """Simple lexical/fuzzy retriever inspired by hybrid metadata+semantic flow."""

    def __init__(self, min_results: int = 1, max_results: int = 8, relative_threshold: float = 0.72) -> None:
        self.min_results = min_results
        self.max_results = max_results
        self.relative_threshold = relative_threshold

    def _score(self, query: QuerySpec, dish: DishDoc) -> float:
        if not query.tokens:
            return 0.0

        overlap = sum(1 for token in query.tokens if token in dish.tokens)
        ratio = overlap / len(set(query.tokens))

        bonus = 0.0
        if normalize_text(dish.dish_name) in normalize_text(query.original_text):
            bonus += 0.6

        return ratio + bonus

    def retrieve(self, query: QuerySpec, docs: list[DishDoc]) -> list[int]:
        scored: list[tuple[float, int]] = []
        for doc in docs:
            score = self._score(query, doc)
            if score > 0:
                scored.append((score, doc.dish_id))

        if not scored:
            return [docs[0].dish_id]

        scored.sort(key=lambda item: (-item[0], item[1]))
        top_score = scored[0][0]

        selected = [dish_id for score, dish_id in scored if score >= top_score * self.relative_threshold][: self.max_results]
        if len(selected) < self.min_results:
            selected = [dish_id for _, dish_id in scored[: self.min_results]]

        return sorted(set(selected))
