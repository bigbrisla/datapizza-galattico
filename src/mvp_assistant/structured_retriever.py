from .models import Recipe, StructuredQuery


class StructuredRetriever:
    def __init__(self, fallback_limit: int = 8) -> None:
        self.fallback_limit = fallback_limit

    @staticmethod
    def _contains(recipe: Recipe, term: str) -> bool:
        if term == recipe.dish_name:
            return True
        if term == recipe.restaurant:
            return True
        if recipe.planet and term == recipe.planet:
            return True
        if term in recipe.ingredients:
            return True
        if term in recipe.techniques:
            return True
        return term in recipe.normalized_text

    def _matches(self, recipe: Recipe, query: StructuredQuery) -> bool:
        if any(not self._contains(recipe, term) for term in query.context_terms):
            return False
        if query.license_name:
            license_names = query.license_name.split("|")
            grade = max((recipe.licenses.get(name, -1) for name in license_names), default=-1)
            if grade < 0:
                grade = None
            if grade is None:
                return False
            if query.license_min_grade is not None and grade < query.license_min_grade:
                return False
        if any(self._contains(recipe, term) for term in query.negative_terms):
            return False
        if query.min_match_count is not None:
            count = sum(1 for term in query.min_match_terms if self._contains(recipe, term))
            if count < query.min_match_count:
                return False

        positives = query.positive_terms
        if not positives:
            return True
        if query.has_or:
            return any(self._contains(recipe, term) for term in positives)
        return all(self._contains(recipe, term) for term in positives)

    def _score(self, recipe: Recipe, query: StructuredQuery) -> float:
        score = 0.0
        for term in query.context_terms:
            if self._contains(recipe, term):
                score += 3.0
        for term in query.positive_terms:
            if self._contains(recipe, term):
                score += 2.0
        for term in query.min_match_terms:
            if self._contains(recipe, term):
                score += 1.0
        if query.license_name and any(name in recipe.licenses for name in query.license_name.split("|")):
            score += 2.0
        return score

    def retrieve(self, query: StructuredQuery, recipes: list[Recipe]) -> list[int]:
        candidates = recipes
        if query.context_terms:
            context_matches = [
                recipe
                for recipe in recipes
                if all(self._contains(recipe, term) for term in query.context_terms)
            ]
            if context_matches:
                candidates = context_matches

        strict_matches = [recipe.dish_id for recipe in candidates if self._matches(recipe, query)]
        if strict_matches:
            has_strong_constraints = bool(query.context_terms) or bool(query.license_name) or query.min_match_count is not None
            if not has_strong_constraints and len(strict_matches) > max(30, int(len(candidates) * 0.6)):
                strict_matches = []
            else:
                return sorted(set(strict_matches))

        scored_matches = [
            (self._score(recipe, query), recipe.dish_id)
            for recipe in candidates
            if self._score(recipe, query) > 0
        ]
        if scored_matches:
            top_score = max(score for score, _ in scored_matches)
            return sorted({dish_id for score, dish_id in scored_matches if score == top_score})

        scored = []
        for recipe in candidates:
            if any(self._contains(recipe, term) for term in query.negative_terms):
                continue
            score = self._score(recipe, query)
            if score > 0:
                scored.append((score, recipe.dish_id))

        if not scored:
            return [recipes[0].dish_id]

        scored.sort(key=lambda item: (-item[0], item[1]))
        return sorted({dish_id for _, dish_id in scored[: self.fallback_limit]})
