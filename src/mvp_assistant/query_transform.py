import os
import re
from pathlib import Path

from .cache import JsonCache
from .models import Recipe, StructuredQuery
from .prompts import query_retry_prompt, query_transform_prompt
from .structured_query import analyze_structured_query, build_phrase_vocabulary
from .text_utils import normalize_text, tokenize


class MetadataQueryTransformer:
    """Query step: transform natural language into a Mongo-style metadata query."""

    def __init__(
        self,
        recipes: list[Recipe],
        questions: list[str],
        llm_client=None,
        difficulty_by_row: dict[int, str] | None = None,
    ) -> None:
        self._llm_client = llm_client
        cache_root = Path(os.getenv("MVP_CACHE_DIR", ".cache/mvp_assistant")) / "query_transform"
        self._cache = JsonCache(cache_root)
        repair_cache_root = Path(os.getenv("MVP_CACHE_DIR", ".cache/mvp_assistant")) / "query_transform_repair"
        self._repair_cache = JsonCache(repair_cache_root)
        self._named_cache_dir = Path(os.getenv("MVP_CACHE_DIR", ".cache/mvp_assistant")) / "query_transform_named"
        self._named_cache_dir.mkdir(parents=True, exist_ok=True)
        self._question_index = {question: idx for idx, question in enumerate(questions, start=1)}
        self._difficulty_by_row = {k: (v or "").strip().lower() for k, v in (difficulty_by_row or {}).items()}
        self._known_vocabulary = self._build_known_vocabulary(recipes)
        self._normalized_vocabulary = {
            field: [normalize_text(value) for value in values if value]
            for field, values in self._known_vocabulary.items()
        }
        self._phrase_vocab = build_phrase_vocabulary(recipes, questions)
        self._generic_terms = self._build_generic_terms(questions)
        self._context_vocab = {
            value
            for recipe in recipes
            for value in [recipe.planet, recipe.restaurant]
            if value
        }
        self._context_vocab = {value.lower() for value in self._context_vocab}

    def transform(self, question: str, row_id: int | None = None) -> StructuredQuery:
        difficulty = self._difficulty_by_row.get(row_id, "")
        heuristic_query = analyze_structured_query(question, self._phrase_vocab, self._context_vocab)
        anchors = self._extract_question_anchors(question)
        llm_query = self._transform_with_llm(question)
        if llm_query and self._is_llm_query_coherent(llm_query, anchors, difficulty):
            merged = self._merge_with_heuristic(llm_query, heuristic_query)
            merged = self._apply_difficulty_policy(merged, difficulty)
            merged = self._constrain_easy_targets(merged, anchors, difficulty)
            merged = self._constrain_medium_context(merged, anchors, difficulty)
            merged.license_name = self._normalize_license_name(merged.license_name)
            self._write_named_query(question, merged, label="final")
            return merged

        repaired_llm = self._repair_with_llm(question, llm_query, anchors)
        if repaired_llm and self._is_llm_query_coherent(repaired_llm, anchors, difficulty):
            merged = self._merge_with_heuristic(repaired_llm, heuristic_query)
            merged = self._apply_difficulty_policy(merged, difficulty)
            merged = self._constrain_easy_targets(merged, anchors, difficulty)
            merged = self._constrain_medium_context(merged, anchors, difficulty)
            merged.license_name = self._normalize_license_name(merged.license_name)
            self._write_named_query(question, merged, label="final")
            return merged

        repaired = self._repair_with_anchors(heuristic_query, anchors)
        repaired = self._apply_difficulty_policy(repaired, difficulty)
        repaired = self._constrain_easy_targets(repaired, anchors, difficulty)
        repaired = self._constrain_medium_context(repaired, anchors, difficulty)
        repaired.license_name = self._normalize_license_name(repaired.license_name)
        self._write_named_query(question, repaired, label="final")
        return repaired

    def _merge_with_heuristic(self, llm_query: StructuredQuery, heuristic_query: StructuredQuery) -> StructuredQuery:
        return StructuredQuery(
            original_text=llm_query.original_text,
            positive_terms=llm_query.positive_terms or heuristic_query.positive_terms,
            negative_terms=llm_query.negative_terms or heuristic_query.negative_terms,
            context_terms=llm_query.context_terms or heuristic_query.context_terms,
            min_match_terms=llm_query.min_match_terms or heuristic_query.min_match_terms,
            min_match_count=llm_query.min_match_count if llm_query.min_match_count is not None else heuristic_query.min_match_count,
            has_or=llm_query.has_or or heuristic_query.has_or,
            license_name=llm_query.license_name or heuristic_query.license_name,
            license_min_grade=(
                llm_query.license_min_grade
                if llm_query.license_min_grade is not None
                else heuristic_query.license_min_grade
            ),
            mongo_query=llm_query.mongo_query,
        )

    def _normalize_license_name(self, license_name: str | None) -> str | None:
        if not license_name:
            return None
        candidates = [normalize_text(value) for value in self._known_vocabulary.get("licenses", []) if value]
        if not candidates:
            return normalize_text(license_name)

        target = normalize_text(license_name)
        if target in candidates:
            return target

        # Generic closest-match by substring containment.
        containing = [candidate for candidate in candidates if target in candidate or candidate in target]
        if containing:
            return sorted(containing, key=len)[0]
        return target

    def _extract_question_anchors(self, question: str) -> dict[str, set[str]]:
        question_norm = normalize_text(question)
        ingredients = {
            normalize_text(value)
            for value in self._known_vocabulary.get("ingredients", [])
            if value and normalize_text(value) in question_norm
        }
        techniques = {
            normalize_text(value)
            for value in self._known_vocabulary.get("techniques", [])
            if value and normalize_text(value) in question_norm
        }
        contexts = {
            normalize_text(value)
            for value in (self._known_vocabulary.get("planet", []) + self._known_vocabulary.get("restaurant", []))
            if value and normalize_text(value) in question_norm
        }
        surface_entities = self._extract_surface_entities(question)
        return {
            "ingredients": ingredients,
            "techniques": techniques,
            "contexts": contexts,
            "surface": surface_entities,
        }

    def _is_llm_query_coherent(self, llm_query: StructuredQuery, anchors: dict[str, set[str]], difficulty: str) -> bool:
        # Reject weak/empty operator payloads that frequently produce noisy broad matches.
        if llm_query.mongo_query and self._has_empty_operator(llm_query.mongo_query):
            return False

        expected_terms = anchors["ingredients"] | anchors["techniques"] | anchors["contexts"] | anchors["surface"]
        if not expected_terms:
            return True

        present = set(llm_query.positive_terms) | set(llm_query.context_terms)
        if llm_query.mongo_query:
            present |= self._collect_query_literals(llm_query.mongo_query)
            if self._has_surface_only_in_dish_name(llm_query.mongo_query, anchors.get("surface", set())):
                return False
            if difficulty == "easy" and not self._easy_fields_are_relevant(llm_query.mongo_query):
                return False
            if difficulty == "easy" and not self._easy_targets_grounded(llm_query.mongo_query, anchors):
                return False

        overlap = len(expected_terms & present)
        # For simple questions with 1-2 anchors we need at least one explicit anchor carried.
        if len(expected_terms) <= 2:
            return overlap >= 1
        # For richer questions ask for stronger alignment.
        return overlap >= 2

    def _easy_fields_are_relevant(self, mongo_query: dict) -> bool:
        values_by_field = self._collect_field_literals(mongo_query)
        # Easy should not rely only on narrative fields.
        has_core = bool(values_by_field.get("ingredients")) or bool(values_by_field.get("techniques"))
        if has_core:
            return True
        return False

    def _easy_targets_grounded(self, mongo_query: dict, anchors: dict[str, set[str]]) -> bool:
        target_terms = set(anchors.get("ingredients", set())) | set(anchors.get("techniques", set())) | set(anchors.get("surface", set()))
        if not target_terms:
            return True
        values_by_field = self._collect_field_literals(mongo_query)
        grounded_values = set(values_by_field.get("ingredients", set())) | set(values_by_field.get("techniques", set()))
        # At least one target anchor should be grounded in ingredients/techniques.
        return len(target_terms & grounded_values) > 0

    def _has_surface_only_in_dish_name(self, mongo_query: dict, surface_terms: set[str]) -> bool:
        if not surface_terms:
            return False
        values_by_field = self._collect_field_literals(mongo_query)
        dish_values = values_by_field.get("dish_name", set())
        ingredient_values = values_by_field.get("ingredients", set())
        technique_values = values_by_field.get("techniques", set())
        for term in surface_terms:
            if term in dish_values and term not in ingredient_values and term not in technique_values:
                return True
        return False

    def _collect_field_literals(self, node, current_field: str | None = None) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        if isinstance(node, dict):
            for key, value in node.items():
                next_field = current_field
                if key in {"dish_name", "ingredients", "techniques", "restaurant", "planet"}:
                    next_field = key
                nested = self._collect_field_literals(value, next_field)
                for field, vals in nested.items():
                    out.setdefault(field, set()).update(vals)
        elif isinstance(node, list):
            for item in node:
                nested = self._collect_field_literals(item, current_field)
                for field, vals in nested.items():
                    out.setdefault(field, set()).update(vals)
        elif isinstance(node, (str, int, float)):
            if current_field:
                out.setdefault(current_field, set()).add(normalize_text(str(node)))
        return out

    def _has_empty_operator(self, node) -> bool:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"$in", "$all", "$or", "$and", "$nor"} and isinstance(value, list) and len(value) == 0:
                    return True
                if self._has_empty_operator(value):
                    return True
        elif isinstance(node, list):
            return any(self._has_empty_operator(item) for item in node)
        return False

    def _collect_query_literals(self, node) -> set[str]:
        values: set[str] = set()
        if isinstance(node, dict):
            for value in node.values():
                values |= self._collect_query_literals(value)
        elif isinstance(node, list):
            for item in node:
                values |= self._collect_query_literals(item)
        elif isinstance(node, (str, int, float)):
            values.add(normalize_text(str(node)))
        return values

    def _repair_with_anchors(self, query: StructuredQuery, anchors: dict[str, set[str]]) -> StructuredQuery:
        anchor_positive = sorted(anchors["ingredients"] | anchors["techniques"] | anchors["surface"])
        anchor_context = sorted(anchors["contexts"])
        promoted = self._promote_structured_terms_from_positive(query.positive_terms)
        if anchor_positive or promoted:
            # Precision-first: when explicit targets are detected in the question,
            # use them as primary positive constraints instead of broad narrative terms.
            positive_terms = sorted(set(anchor_positive) | set(promoted))
        else:
            positive_terms = sorted(set(query.positive_terms) | set(anchor_positive))
        context_terms = sorted(set(query.context_terms) | set(anchor_context))

        return StructuredQuery(
            original_text=query.original_text,
            positive_terms=positive_terms,
            negative_terms=query.negative_terms,
            context_terms=context_terms,
            min_match_terms=query.min_match_terms,
            min_match_count=query.min_match_count,
            has_or=query.has_or,
            license_name=query.license_name,
            license_min_grade=query.license_min_grade,
            # Drop incoherent mongo draft to avoid broad/noisy retrieval.
            mongo_query=None,
        )

    def _extract_surface_entities(self, question: str) -> set[str]:
        # Generic extraction of salient entity-like spans (e.g., "Cioccorane", "Latte+")
        # without domain-specific lists.
        entities: set[str] = set()
        generic = set(self._generic_terms)
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9\+\-']{2,}(?:\s+[A-Z][A-Za-z0-9\+\-']{2,}){0,3}\b", question):
            value = normalize_text(match.group(0))
            if value:
                if value in generic:
                    continue
                entities.add(value)
        return entities

    def _promote_structured_terms_from_positive(self, positive_terms: list[str]) -> list[str]:
        vocab_ingredients = set(self._normalized_vocabulary.get("ingredients", []))
        vocab_techniques = set(self._normalized_vocabulary.get("techniques", []))
        vocab_structured = vocab_ingredients | vocab_techniques
        promoted: set[str] = set()

        for term in positive_terms:
            term_norm = normalize_text(term)
            if not term_norm:
                continue
            if term_norm in vocab_structured:
                promoted.add(term_norm)
                continue
            # Phrase to tokens: keep only tokens that map to structured vocab.
            for tok in term_norm.split():
                if tok in vocab_structured:
                    promoted.add(tok)
            # Phrase containment against vocab entries (generic, no domain constants).
            for value in vocab_structured:
                if value in term_norm or term_norm in value:
                    promoted.add(value)
        return sorted(promoted)



    def _build_known_vocabulary(self, recipes: list[Recipe]) -> dict[str, list[str]]:
        vocabulary = {
            "dish_name": set(),
            "restaurant": set(),
            "planet": set(),
            "ingredients": set(),
            "techniques": set(),
            "licenses": set(),
        }
        for recipe in recipes:
            vocabulary["dish_name"].add(recipe.dish_name)
            if recipe.restaurant:
                vocabulary["restaurant"].add(recipe.restaurant)
            if recipe.planet:
                vocabulary["planet"].add(recipe.planet)
            vocabulary["ingredients"].update(recipe.ingredients)
            vocabulary["techniques"].update(recipe.techniques)
            vocabulary["licenses"].update(recipe.licenses)
        return {field: sorted(values) for field, values in vocabulary.items()}

    def _transform_with_llm(self, question: str) -> StructuredQuery | None:
        if self._llm_client is None:
            return None
        if os.getenv("MVP_ENABLE_LLM_QUERY", "1") != "1":
            return None

        row_id = self._question_index.get(question)
        difficulty = self._difficulty_by_row.get(row_id, "")
        allowed_fields = self._allowed_fields_for_difficulty(difficulty)
        prompt = query_transform_prompt(
            question=question,
            available_fields=list(self._known_vocabulary),
            known_vocabulary=self._known_vocabulary,
            allowed_fields=allowed_fields,
            generic_terms_to_avoid=self._generic_terms,
        )
        cache_key = self._cache.key_for(question)
        payload = self._cache.get(cache_key)
        if payload is None:
            payload = self._llm_client.complete_json(prompt)
            if payload:
                self._cache.set(cache_key, payload)
                self._write_named_payload(question, payload, prefix="query_metadata")
        if not payload:
            return None

        return StructuredQuery(
            original_text=question,
            positive_terms=[normalize_text(item) for item in payload.get("positive_terms", []) if item],
            negative_terms=[normalize_text(item) for item in payload.get("negative_terms", []) if item],
            context_terms=[normalize_text(item) for item in payload.get("context_terms", []) if item],
            min_match_terms=[normalize_text(item) for item in payload.get("min_match_terms", []) if item],
            min_match_count=payload.get("min_match_count"),
            has_or=bool(payload.get("has_or", False)),
            license_name=payload.get("license_name"),
            license_min_grade=payload.get("license_min_grade"),
            mongo_query=payload.get("query") if isinstance(payload.get("query"), dict) else None,
        )

    def _repair_with_llm(
        self,
        question: str,
        previous_query: StructuredQuery | None,
        anchors: dict[str, set[str]],
    ) -> StructuredQuery | None:
        if self._llm_client is None:
            return None
        if os.getenv("MVP_ENABLE_LLM_QUERY", "1") != "1":
            return None

        previous_mongo = previous_query.mongo_query if previous_query and previous_query.mongo_query else {}
        anchor_terms = sorted(anchors["ingredients"] | anchors["techniques"] | anchors["contexts"])
        reason = (
            "La query deve mantenere i vincoli espliciti della domanda sui termini: "
            + ", ".join(anchor_terms)
            if anchor_terms
            else "La query precedente e incoerente o troppo generica."
        )
        prompt = query_retry_prompt(
            question=question,
            previous_query=previous_mongo,
            reason=reason,
            known_vocabulary=self._known_vocabulary,
        )
        cache_key = self._repair_cache.key_for(question)
        payload = self._repair_cache.get(cache_key)
        if payload is None:
            payload = self._llm_client.complete_json(prompt)
            if payload:
                self._repair_cache.set(cache_key, payload)
                self._write_named_payload(question, payload, prefix="query_metadata_repair")
        if not payload:
            return None
        return StructuredQuery(
            original_text=question,
            positive_terms=[normalize_text(item) for item in payload.get("positive_terms", []) if item],
            negative_terms=[normalize_text(item) for item in payload.get("negative_terms", []) if item],
            context_terms=[normalize_text(item) for item in payload.get("context_terms", []) if item],
            min_match_terms=[normalize_text(item) for item in payload.get("min_match_terms", []) if item],
            min_match_count=payload.get("min_match_count"),
            has_or=bool(payload.get("has_or", False)),
            license_name=payload.get("license_name"),
            license_min_grade=payload.get("license_min_grade"),
            mongo_query=payload.get("query") if isinstance(payload.get("query"), dict) else None,
        )

    def _allowed_fields_for_difficulty(self, difficulty: str) -> list[str]:
        if difficulty == "easy":
            return ["ingredients", "techniques"]
        if difficulty == "medium":
            return ["ingredients", "techniques", "planet", "planet_name", "restaurant", "restaurant_name", "licenses", "licences"]
        return list(self._known_vocabulary)

    def _build_generic_terms(self, questions: list[str]) -> list[str]:
        from collections import Counter

        vocab_tokens: set[str] = set()
        for values in self._known_vocabulary.values():
            for value in values:
                vocab_tokens.update(tokenize(value))

        counter = Counter()
        for question in questions:
            counter.update(tokenize(question))

        threshold = max(4, len(questions) // 25)
        generic = [
            term
            for term, freq in counter.items()
            if freq >= threshold and term not in vocab_tokens and len(term) > 2
        ]
        return sorted(generic)[:80]

    def _write_named_payload(self, question: str, payload: dict, prefix: str) -> None:
        row_id = self._question_index.get(question)
        if row_id is None:
            return
        path = self._named_cache_dir / f"{prefix}_{row_id}.json"
        path.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_named_query(self, question: str, query: StructuredQuery, label: str) -> None:
        row_id = self._question_index.get(question)
        if row_id is None:
            return
        payload = {
            "row_id": row_id,
            "label": label,
            "positive_terms": query.positive_terms,
            "negative_terms": query.negative_terms,
            "context_terms": query.context_terms,
            "min_match_terms": query.min_match_terms,
            "min_match_count": query.min_match_count,
            "has_or": query.has_or,
            "license_name": query.license_name,
            "license_min_grade": query.license_min_grade,
            "query": query.mongo_query,
        }
        path = self._named_cache_dir / f"query_metadata_{row_id}.json"
        path.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _apply_difficulty_policy(self, query: StructuredQuery, difficulty: str) -> StructuredQuery:
        if difficulty == "easy":
            # Easy questions should hinge on ingredients/techniques only.
            return StructuredQuery(
                original_text=query.original_text,
                positive_terms=query.positive_terms,
                negative_terms=query.negative_terms,
                context_terms=[],
                min_match_terms=query.min_match_terms,
                min_match_count=query.min_match_count,
                has_or=query.has_or,
                license_name=None,
                license_min_grade=None,
                mongo_query=self._strip_query_fields(query.mongo_query, {"ingredients", "techniques"}),
            )
        if difficulty == "medium":
            return StructuredQuery(
                original_text=query.original_text,
                positive_terms=query.positive_terms,
                negative_terms=query.negative_terms,
                context_terms=query.context_terms,
                min_match_terms=query.min_match_terms,
                min_match_count=query.min_match_count,
                has_or=query.has_or,
                license_name=query.license_name,
                license_min_grade=query.license_min_grade,
                mongo_query=self._strip_query_fields(
                    query.mongo_query,
                    {"ingredients", "techniques", "planet", "planet_name", "restaurant", "restaurant_name", "licenses", "licences"},
                ),
            )
        return query

    def _constrain_easy_targets(
        self,
        query: StructuredQuery,
        anchors: dict[str, set[str]],
        difficulty: str,
    ) -> StructuredQuery:
        if difficulty != "easy":
            return query
        anchored_targets = sorted(
            set(anchors.get("ingredients", set()))
            | set(anchors.get("techniques", set()))
            | set(anchors.get("surface", set()))
        )
        if not anchored_targets:
            return query
        return StructuredQuery(
            original_text=query.original_text,
            positive_terms=anchored_targets,
            negative_terms=query.negative_terms,
            context_terms=[],
            min_match_terms=query.min_match_terms,
            min_match_count=query.min_match_count,
            has_or=query.has_or,
            license_name=None,
            license_min_grade=None,
            mongo_query=query.mongo_query,
        )

    def _constrain_medium_context(
        self,
        query: StructuredQuery,
        anchors: dict[str, set[str]],
        difficulty: str,
    ) -> StructuredQuery:
        if difficulty != "medium":
            return query
        context = sorted(set(query.context_terms) | set(anchors.get("contexts", set())))
        return StructuredQuery(
            original_text=query.original_text,
            positive_terms=query.positive_terms,
            negative_terms=query.negative_terms,
            context_terms=context,
            min_match_terms=query.min_match_terms,
            min_match_count=query.min_match_count,
            has_or=query.has_or,
            license_name=query.license_name,
            license_min_grade=query.license_min_grade,
            mongo_query=query.mongo_query,
        )

    def _strip_query_fields(self, mongo_query: dict | None, allowed_fields: set[str]) -> dict | None:
        if not mongo_query:
            return None

        def recurse(node):
            if isinstance(node, dict):
                out = {}
                for k, v in node.items():
                    if k in {"$and", "$or", "$nor"} and isinstance(v, list):
                        items = [recurse(item) for item in v]
                        items = [item for item in items if item]
                        if items:
                            out[k] = items
                        continue
                    if k.startswith("$"):
                        out[k] = recurse(v)
                        continue
                    if k in allowed_fields:
                        vv = recurse(v)
                        if vv is not None and vv != {} and vv != []:
                            out[k] = vv
                return out
            if isinstance(node, list):
                items = [recurse(item) for item in node]
                return [item for item in items if item]
            return node

        cleaned = recurse(mongo_query)
        return cleaned if cleaned else None
