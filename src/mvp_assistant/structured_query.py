from .models import Recipe, StructuredQuery
from .text_utils import normalize_text
import re

def _question_ngrams(question: str, max_words: int = 8) -> set[str]:
    words = normalize_text(question).split()
    terms: set[str] = set()
    for size in range(1, max_words + 1):
        if len(words) < size:
            continue
        for start in range(0, len(words) - size + 1):
            phrase = " ".join(words[start : start + size])
            if len(phrase) >= 4:
                terms.add(phrase)
    return terms


def _metadata_terms(recipe: Recipe) -> set[str]:
    terms = {
        recipe.dish_name,
        recipe.restaurant,
        recipe.planet or "",
        *recipe.ingredients,
        *recipe.techniques,
    }
    return {normalize_text(term) for term in terms if term}


def _metadata_token_support(metadata_terms: set[str]) -> set[str]:
    tokens: set[str] = set()
    for term in metadata_terms:
        tokens.update(term.split())
    return tokens


def _supported_by_metadata(term: str, metadata_terms: set[str], metadata_tokens: set[str]) -> bool:
    if term in metadata_terms:
        return True
    parts = term.split()
    if not parts:
        return False
    if len(parts) == 1:
        return term in metadata_tokens
    return any(term in metadata_term for metadata_term in metadata_terms)


def build_phrase_vocabulary(recipes: list[Recipe], questions: list[str]) -> set[str]:
    recipe_terms = {term for recipe in recipes for term in _metadata_terms(recipe)}
    recipe_tokens = _metadata_token_support(recipe_terms)
    question_terms = set().union(*(_question_ngrams(question) for question in questions))
    supported_terms = [
        term
        for term in question_terms
        if _supported_by_metadata(term, recipe_terms, recipe_tokens)
    ]
    single_word_terms = [term for term in supported_terms if len(term.split()) == 1]
    multi_word_terms = [term for term in supported_terms if len(term.split()) > 1]

    def doc_freq(term: str) -> int:
        return sum(1 for recipe in recipes if term in recipe.normalized_text)

    single_word_freqs = {term: doc_freq(term) for term in single_word_terms}
    if single_word_freqs:
        ordered = sorted(single_word_freqs.values())
        median = ordered[len(ordered) // 2]
        selected_single = {term for term, freq in single_word_freqs.items() if freq <= median}
    else:
        selected_single = set()

    discovered_terms = set(multi_word_terms) | selected_single
    return recipe_terms | discovered_terms


def _dedupe_contained(terms: list[str]) -> list[str]:
    selected: list[str] = []
    for term in sorted(set(terms), key=lambda value: (-len(value), value)):
        if not any(term in existing for existing in selected):
            selected.append(term)
    return sorted(selected)


def _mentions_or(question_norm: str) -> bool:
    return any(marker in question_norm for marker in (" o ", " oppure "))


def _extract_min_match_count(question_norm: str) -> int | None:
    match = re.search(r"almeno\s+(\d+)\s+ingredient", question_norm)
    if match:
        return int(match.group(1))
    return None


def _extract_license_constraint(question_norm: str) -> tuple[str | None, int | None]:
    # Generic: capture license label and optional numeric threshold around "grado"/comparative forms.
    match = re.search(
        r"licenz\w*\s+([a-z0-9\+\-_]+)(?:\s+\w+){0,6}?(?:grado\s*)?(?:superiore|maggiore|oltre|>=|>)\s*(\d+)",
        question_norm,
    )
    if match:
        return match.group(1), int(match.group(2)) + (1 if match.group(0).find(">") != -1 and ">=" not in match.group(0) else 0)

    match = re.search(r"licenz\w*\s+([a-z0-9\+\-_]+)(?:\s+\w+){0,4}?grado\s+(\d+)", question_norm)
    if match:
        return match.group(1), int(match.group(2))

    match = re.search(r"licenz\w*\s+([a-z0-9\+\-_]+)", question_norm)
    if match:
        return match.group(1), None
    return None, None


def analyze_structured_query(question: str, phrase_vocab: set[str], context_vocab: set[str] | None = None) -> StructuredQuery:
    question_norm = normalize_text(question)
    context_vocab = context_vocab or set()

    matched_terms = [term for term in phrase_vocab if term and term in question_norm]
    context_terms = [term for term in matched_terms if term in context_vocab]
    positive_terms = [term for term in matched_terms if term not in context_vocab]
    negative_terms: list[str] = []
    min_match_count = _extract_min_match_count(question_norm)
    license_name, license_min_grade = _extract_license_constraint(question_norm)

    return StructuredQuery(
        original_text=question,
        positive_terms=_dedupe_contained(positive_terms),
        negative_terms=_dedupe_contained(negative_terms),
        context_terms=_dedupe_contained(context_terms),
        min_match_terms=_dedupe_contained(positive_terms) if min_match_count is not None else [],
        min_match_count=min_match_count,
        has_or=_mentions_or(f" {question_norm} "),
        license_name=license_name,
        license_min_grade=license_min_grade,
    )
