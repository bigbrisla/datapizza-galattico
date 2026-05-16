from .models import DishDoc
from .text_utils import normalize_text, tokenize


class DishIndexBuilder:
    """Builds dish-centric pseudo-chunks from the mixed knowledge base corpus."""

    def __init__(self, context_window: int = 1600) -> None:
        self.context_window = context_window

    def build(self, corpus: str, dish_mapping: dict[str, int]) -> list[DishDoc]:
        norm_corpus = normalize_text(corpus)
        docs: list[DishDoc] = []

        for dish_name, dish_id in dish_mapping.items():
            norm_name = normalize_text(dish_name)
            pos = norm_corpus.find(norm_name)

            if pos >= 0:
                start = max(0, pos - self.context_window)
                end = min(len(corpus), pos + len(dish_name) + self.context_window)
                local_context = corpus[start:end]
            else:
                local_context = ""

            combined = f"{dish_name}\n{local_context}\n"
            docs.append(
                DishDoc(
                    dish_id=dish_id,
                    dish_name=dish_name,
                    text=combined,
                    tokens=set(tokenize(combined)),
                )
            )

        return docs
