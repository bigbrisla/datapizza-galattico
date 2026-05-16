import os
from pathlib import Path

from .cache import JsonCache
from .prompts import candidate_rerank_prompt
from .models import Recipe, StructuredQuery
from .structured_retriever import StructuredRetriever
from .text_utils import normalize_text


class RecipeMetadataStore:
    """MongoDB-like in-memory collection used as primary retrieval store."""

    def __init__(self, llm_client=None) -> None:
        self._records: list[Recipe] = []
        self._retriever = StructuredRetriever()
        self._llm_client = llm_client
        cache_root = Path(os.getenv("MVP_CACHE_DIR", ".cache/mvp_assistant")) / "answer_rerank"
        self._cache = JsonCache(cache_root)

    def insert_many(self, recipes: list[Recipe]) -> None:
        self._records.extend(recipes)

    def all(self) -> list[Recipe]:
        return list(self._records)

    def find(self, query: StructuredQuery) -> list[int]:
        strict_field_results = self._find_strict_field_match(query)
        if strict_field_results:
            return strict_field_results

        deterministic_results, candidate_ids = self._find_deterministic(query)
        if self._is_confident(query, deterministic_results):
            return self._shrink_if_too_broad(query, deterministic_results)

        reranked = self._find_with_llm(query.original_text, candidate_ids)
        if reranked:
            return reranked

        return self._shrink_if_too_broad(query, deterministic_results)

    def _find_strict_field_match(self, query: StructuredQuery) -> list[int]:
        # Precision-first path for simple target queries:
        # avoid broad text fallback and require exact metadata field membership.
        if len(query.positive_terms) != 1:
            return []
        if query.context_terms or query.negative_terms:
            return []
        if query.license_name or query.license_min_grade is not None:
            return []
        if query.min_match_count is not None or query.min_match_terms:
            return []

        term_norm = normalize_text(query.positive_terms[0])
        if not term_norm:
            return []
        if not any(term_norm in record.ingredients or term_norm in record.techniques for record in self._records):
            return []
        matched = [
            record.dish_id
            for record in self._records
            if term_norm in record.ingredients or term_norm in record.techniques
        ]
        return sorted(set(matched))

    def _find_deterministic(self, query: StructuredQuery) -> tuple[list[int], list[int]]:
        if query.mongo_query:
            mongo_results = self._find_mongo_style(query.mongo_query)
            if mongo_results:
                constrained = self._apply_structured_constraints(query, mongo_results)
                if constrained:
                    return constrained, mongo_results

        results = self._retriever.retrieve(query, self._records)
        if results:
            return results, results

        if not self._should_relax(query):
            return [], []

        relaxed = StructuredQuery(
            original_text=query.original_text,
            positive_terms=query.positive_terms[: max(1, len(query.positive_terms) // 2)],
            negative_terms=query.negative_terms,
            context_terms=query.context_terms,
            min_match_terms=[],
            min_match_count=None,
            has_or=True,
            license_name=query.license_name,
            license_min_grade=query.license_min_grade,
        )
        relaxed_results = self._retriever.retrieve(relaxed, self._records)
        return relaxed_results, relaxed_results

    def _should_relax(self, query: StructuredQuery) -> bool:
        # Relax only when the question is inherently vague. For specific constraints,
        # broad fallback tends to inflate false positives.
        if query.license_name or query.license_min_grade is not None:
            return False
        if query.context_terms:
            return False
        if query.negative_terms:
            return False
        if query.min_match_count is not None or query.min_match_terms:
            return False
        if len(query.positive_terms) >= 2:
            return False
        return True

    def _apply_structured_constraints(self, query: StructuredQuery, candidate_ids: list[int]) -> list[int]:
        has_constraints = any(
            [
                query.context_terms,
                query.positive_terms,
                query.negative_terms,
                query.min_match_terms,
                query.min_match_count is not None,
                query.license_name,
                query.license_min_grade is not None,
            ]
        )
        if not has_constraints:
            return candidate_ids
        candidate_set = set(candidate_ids)
        candidate_records = [record for record in self._records if record.dish_id in candidate_set]
        return self._retriever.retrieve(query, candidate_records)

    def _is_confident(self, query: StructuredQuery, results: list[int]) -> bool:
        if not results:
            return False
        if len(results) > 20:
            return False
        if len(results) == len(self._records):
            return False

        has_strong_constraints = any(
            [
                bool(query.context_terms),
                bool(query.positive_terms),
                bool(query.license_name),
                query.min_match_count is not None,
                bool(query.mongo_query),
            ]
        )
        return has_strong_constraints

    def _query_specificity(self, query: StructuredQuery) -> int:
        specificity = 0
        specificity += len(query.context_terms) * 3
        specificity += len(query.positive_terms) * 2
        specificity += len(query.min_match_terms)
        if query.min_match_count is not None:
            specificity += 2
        if query.license_name:
            specificity += 3
            if query.license_min_grade is not None:
                specificity += 2
        if query.mongo_query:
            specificity += 2
        return specificity

    def _shrink_if_too_broad(self, query: StructuredQuery, results: list[int]) -> list[int]:
        if not results:
            return results
        max_results = int(os.getenv("MVP_MAX_RESULTS_PER_QUERY", "12"))
        if len(results) <= max_results:
            return results

        specificity = self._query_specificity(query)
        # For low-specificity queries, very large result sets are likely uninformative.
        if specificity <= 2:
            return results[:max_results]

        # For medium/high-specificity queries keep a bounded output to avoid "match-all" artifacts.
        return results[: max_results * 2]

    def _find_with_llm(self, question: str, candidate_ids: list[int]) -> list[int]:
        if self._llm_client is None:
            return []
        if os.getenv("MVP_ENABLE_LLM_MATCHING", "1") != "1":
            return []
        if not candidate_ids:
            return []

        candidates = [self._record_to_candidate(record) for record in self._records if record.dish_id in set(candidate_ids)]
        if not candidates:
            return []
        if len(candidates) > 40:
            candidates = candidates[:40]

        cache_key = self._cache.key_for(question + "|" + ",".join(str(item["dish_id"]) for item in candidates))
        payload = self._cache.get(cache_key)
        if payload is None:
            prompt = candidate_rerank_prompt(question, candidates)
            payload = self._llm_client.complete_json(prompt)
            if payload:
                self._cache.set(cache_key, payload)
        if not payload:
            return []

        dish_ids = payload.get("dish_ids", [])
        if not isinstance(dish_ids, list):
            return []
        allowed_ids = {item["dish_id"] for item in candidates}
        cleaned = sorted({int(dish_id) for dish_id in dish_ids if isinstance(dish_id, int) and dish_id in allowed_ids})
        return cleaned

    def _record_to_candidate(self, record: Recipe) -> dict:
        return {
            "dish_id": record.dish_id,
            "dish_name": record.dish_name,
            "restaurant": record.restaurant,
            "planet": record.planet,
            "ingredients": sorted(record.ingredients),
            "techniques": sorted(record.techniques),
            "licenses": record.licenses,
        }

    def describe_collection(self) -> dict[str, list[str]]:
        values = {
            "dish_name": set(),
            "restaurant": set(),
            "planet": set(),
            "ingredients": set(),
            "techniques": set(),
            "licenses": set(),
        }
        for record in self._records:
            values["dish_name"].add(record.dish_name)
            if record.restaurant:
                values["restaurant"].add(record.restaurant)
            if record.planet:
                values["planet"].add(record.planet)
            values["ingredients"].update(record.ingredients)
            values["techniques"].update(record.techniques)
            values["licenses"].update(record.licenses)
        return {key: sorted(item for item in value if item) for key, value in values.items()}

    def _find_mongo_style(self, query: dict) -> list[int]:
        matched = [record.dish_id for record in self._records if self._record_matches(record, query)]
        return sorted(set(matched))

    def _record_matches(self, record: Recipe, query: dict) -> bool:
        for field, expected in query.items():
            if field == "$and" and isinstance(expected, list):
                if not all(self._record_matches(record, subquery) for subquery in expected):
                    return False
                continue
            if field == "$or" and isinstance(expected, list):
                if not any(self._record_matches(record, subquery) for subquery in expected):
                    return False
                continue
            if field == "$nor" and isinstance(expected, list):
                if any(self._record_matches(record, subquery) for subquery in expected):
                    return False
                continue
            if not self._field_matches(record, field, expected):
                return False
        return True

    def _field_matches(self, record: Recipe, field: str, expected) -> bool:
        value = self._field_value(record, field)
        if field in {"licenses", "licences"} and isinstance(value, dict):
            return self._license_matches(value, expected)
        if isinstance(expected, dict):
            if "$in" in expected:
                return any(self._value_contains(value, item) for item in expected["$in"])
            if "$all" in expected:
                return all(self._value_contains(value, item) for item in expected["$all"])
            if "$ne" in expected:
                return not self._value_contains(value, expected["$ne"])
            if "$regex" in expected:
                return self._value_contains(value, expected["$regex"])
            return False
        return self._value_contains(value, expected)

    def _field_value(self, record: Recipe, field: str):
        aliases = {
            "dish_name": record.dish_name,
            "restaurant": record.restaurant,
            "restaurant_name": record.restaurant,
            "planet": record.planet,
            "planet_name": record.planet,
            "ingredients": record.ingredients,
            "dish_ingredients": record.ingredients,
            "techniques": record.techniques,
            "dish_techniques": record.techniques,
            "licenses": record.licenses,
            "licences": record.licenses,
        }
        return aliases.get(field)

    def _license_matches(self, licenses: dict[str, int], expected) -> bool:
        if isinstance(expected, str):
            return self._value_contains(set(licenses), expected)
        if not isinstance(expected, dict):
            return False

        if "$in" in expected:
            items = expected["$in"]
            if not isinstance(items, list):
                return False
            return any(self._license_matches(licenses, item) for item in items)

        if "$all" in expected:
            items = expected["$all"]
            if not isinstance(items, list):
                return False
            return all(self._license_matches(licenses, item) for item in items)

        # Generic operator form: {"license_name": {"$gte": 1}}
        for license_name, rule in expected.items():
            if not isinstance(license_name, str):
                continue
            grade = licenses.get(normalize_text(license_name))
            if grade is None:
                return False
            if isinstance(rule, dict):
                if "$gte" in rule and grade < int(rule["$gte"]):
                    return False
                if "$gt" in rule and grade <= int(rule["$gt"]):
                    return False
                if "$lte" in rule and grade > int(rule["$lte"]):
                    return False
                if "$lt" in rule and grade >= int(rule["$lt"]):
                    return False
                if "$eq" in rule and grade != int(rule["$eq"]):
                    return False
            elif isinstance(rule, int):
                if grade != rule:
                    return False
            else:
                return False
        return True

    def _value_contains(self, value, expected) -> bool:
        expected_norm = normalize_text(str(expected))
        if value is None:
            return False
        if isinstance(value, (set, list, tuple)):
            return any(expected_norm in normalize_text(str(item)) for item in value)
        return expected_norm in normalize_text(str(value))
