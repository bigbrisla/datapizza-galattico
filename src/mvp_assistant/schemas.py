from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExtractedIngredient:
    name: str
    quantity: str | None = None
    unit: str | None = None


@dataclass
class ExtractedLicense:
    name: str
    grade: int | None = None
    original_text: str | None = None


@dataclass
class ExtractedRecipe:
    dish_name: str
    restaurant: str | None = None
    planet: str | None = None
    source_document: str | None = None
    ingredients: list[ExtractedIngredient] = field(default_factory=list)
    techniques: list[str] = field(default_factory=list)
    licenses: list[ExtractedLicense] = field(default_factory=list)
    original_text: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recipes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dish_name": {"type": "string"},
                    "restaurant": {"type": ["string", "null"]},
                    "planet": {"type": ["string", "null"]},
                    "source_document": {"type": ["string", "null"]},
                    "ingredients": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "quantity": {"type": ["string", "null"]},
                                "unit": {"type": ["string", "null"]},
                            },
                            "required": ["name"],
                        },
                    },
                    "techniques": {"type": "array", "items": {"type": "string"}},
                    "licenses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "grade": {"type": ["integer", "null"]},
                                "original_text": {"type": ["string", "null"]},
                            },
                            "required": ["name"],
                        },
                    },
                    "original_text": {"type": ["string", "null"]},
                },
                "required": ["dish_name"],
            },
        }
    },
    "required": ["recipes"],
}


MONGO_QUERY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "object",
            "description": "MongoDB-style query over metadata fields.",
        }
    },
    "required": ["query"],
}
