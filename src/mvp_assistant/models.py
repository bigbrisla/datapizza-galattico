from dataclasses import dataclass


@dataclass
class DishDoc:
    dish_id: int
    dish_name: str
    text: str
    tokens: set[str]


@dataclass
class QuerySpec:
    original_text: str
    tokens: list[str]


@dataclass
class Recipe:
    dish_id: int
    dish_name: str
    restaurant: str
    planet: str | None
    source_document: str
    text: str
    normalized_text: str
    ingredients: set[str]
    techniques: set[str]
    licenses: dict[str, int]


@dataclass
class SourceDocument:
    source_document: str
    text: str


@dataclass
class ParsedMenu:
    source_document: str
    text: str
    restaurant: str
    planet: str | None
    licenses: dict[str, int]


@dataclass
class SemanticChunk:
    dish_id: int
    dish_name: str
    source_document: str
    restaurant: str
    planet: str | None
    licenses: dict[str, int]
    text: str


@dataclass
class StructuredQuery:
    original_text: str
    positive_terms: list[str]
    negative_terms: list[str]
    context_terms: list[str]
    min_match_terms: list[str]
    min_match_count: int | None = None
    has_or: bool = False
    license_name: str | None = None
    license_min_grade: int | None = None
    mongo_query: dict | None = None
