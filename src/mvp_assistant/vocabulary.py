from collections import defaultdict

from .schemas import ExtractedRecipe
from .text_utils import normalize_text


class Vocabulary:
    def __init__(self) -> None:
        self._values: dict[str, set[str]] = defaultdict(set)

    def add(self, field: str, value: str | None) -> None:
        if value:
            self._values[field].add(normalize_text(value))

    def update_from_recipe(self, recipe: ExtractedRecipe) -> None:
        self.add("dish_name", recipe.dish_name)
        self.add("restaurant", recipe.restaurant)
        self.add("planet", recipe.planet)
        for ingredient in recipe.ingredients:
            self.add("ingredients.name", ingredient.name)
        for technique in recipe.techniques:
            self.add("techniques", technique)
        for license_item in recipe.licenses:
            self.add("licenses.name", license_item.name)

    def as_prompt_dict(self) -> dict[str, list[str]]:
        return {field: sorted(values) for field, values in self._values.items()}
