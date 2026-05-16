import os
import json
from pathlib import Path

from .cache import JsonCache
from .models import Recipe, SemanticChunk
from .prompts import metadata_extraction_prompt
from .schemas import ExtractedIngredient, ExtractedLicense, ExtractedRecipe
from .text_utils import normalize_text
from .vocabulary import Vocabulary


class RecipeMetadataExtractor:
    """Metadata extraction step: convert dish chunks into queryable recipe records."""

    def __init__(self, llm_client=None, vocabulary: Vocabulary | None = None) -> None:
        self.llm_client = llm_client
        self.vocabulary = vocabulary or Vocabulary()
        cache_root = Path(os.getenv("MVP_CACHE_DIR", ".cache/mvp_assistant")) / "metadata_extraction"
        self.cache = JsonCache(cache_root)
        self.named_cache_dir = Path(os.getenv("MVP_CACHE_DIR", ".cache/mvp_assistant")) / "metadata_extraction_named"
        self.named_cache_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, chunk: SemanticChunk) -> Recipe:
        extracted = self._extract_with_llm(chunk)
        if extracted:
            self.vocabulary.update_from_recipe(extracted)
            recipe = self._recipe_from_extracted(chunk, extracted)
            self._write_named_metadata(chunk, extracted)
            return recipe

        combined = "\n".join([chunk.restaurant, chunk.planet or "", chunk.text])

        return Recipe(
            dish_id=chunk.dish_id,
            dish_name=chunk.dish_name,
            restaurant=chunk.restaurant,
            planet=chunk.planet,
            source_document=chunk.source_document,
            text=combined,
            normalized_text=normalize_text(combined),
            ingredients=set(),
            techniques=set(),
            licenses=chunk.licenses,
        )

    def _write_named_metadata(self, chunk: SemanticChunk, extracted: ExtractedRecipe) -> None:
        payload = {
            "dish_id": chunk.dish_id,
            "dish_name": chunk.dish_name,
            "source_document": chunk.source_document,
            "restaurant": extracted.restaurant or chunk.restaurant,
            "planet": extracted.planet or chunk.planet,
            "ingredients": [item.name for item in extracted.ingredients if item.name],
            "techniques": [item for item in extracted.techniques if item],
            "licenses": [
                {"name": item.name, "grade": item.grade, "original_text": item.original_text}
                for item in extracted.licenses
                if item.name
            ],
            "original_text": extracted.original_text or chunk.text,
        }
        path = self.named_cache_dir / f"text_metadata_{chunk.dish_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _extract_with_llm(self, chunk: SemanticChunk) -> ExtractedRecipe | None:
        if self.llm_client is None:
            return None
        if os.getenv("MVP_ENABLE_LLM_METADATA", "1") != "1":
            return None

        prompt = metadata_extraction_prompt(
            source_document=chunk.source_document,
            chunk_text=chunk.text,
            known_vocabulary=self.vocabulary.as_prompt_dict(),
        )
        cache_key = self.cache.key_for(f"{chunk.source_document}:{chunk.dish_id}:{chunk.text}")
        payload = self.cache.get(cache_key)
        if payload is None:
            payload = self.llm_client.complete_json(prompt)
            if payload:
                self.cache.set(cache_key, payload)
        if not payload:
            return None

        recipes = payload.get("recipes") or []
        if not recipes:
            return None

        raw = recipes[0]
        return ExtractedRecipe(
            dish_name=raw.get("dish_name") or chunk.dish_name,
            restaurant=raw.get("restaurant") or chunk.restaurant,
            planet=raw.get("planet") or chunk.planet,
            source_document=raw.get("source_document") or chunk.source_document,
            ingredients=[
                ExtractedIngredient(
                    name=item.get("name", ""),
                    quantity=item.get("quantity"),
                    unit=item.get("unit"),
                )
                for item in raw.get("ingredients", [])
                if item.get("name")
            ],
            techniques=[item for item in raw.get("techniques", []) if item],
            licenses=[
                ExtractedLicense(
                    name=item.get("name", ""),
                    grade=item.get("grade"),
                    original_text=item.get("original_text"),
                )
                for item in raw.get("licenses", [])
                if item.get("name")
            ],
            original_text=raw.get("original_text") or chunk.text,
        )

    def _recipe_from_extracted(self, chunk: SemanticChunk, extracted: ExtractedRecipe) -> Recipe:
        combined = "\n".join(
            [
                extracted.restaurant or chunk.restaurant,
                extracted.planet or chunk.planet or "",
                extracted.original_text or chunk.text,
            ]
        )
        licenses = {
            normalize_text(item.name): item.grade
            for item in extracted.licenses
            if item.name and item.grade is not None
        }
        for name, grade in chunk.licenses.items():
            existing = licenses.get(name)
            if existing is None or grade > existing:
                licenses[name] = grade
        return Recipe(
            dish_id=chunk.dish_id,
            dish_name=extracted.dish_name or chunk.dish_name,
            restaurant=extracted.restaurant or chunk.restaurant,
            planet=extracted.planet or chunk.planet,
            source_document=extracted.source_document or chunk.source_document,
            text=combined,
            normalized_text=normalize_text(combined),
            ingredients={normalize_text(item.name) for item in extracted.ingredients if item.name},
            techniques={normalize_text(item) for item in extracted.techniques if item},
            licenses=licenses,
        )

    def placeholder(self, dish_name: str, dish_id: int) -> Recipe:
        return Recipe(
            dish_id=dish_id,
            dish_name=dish_name,
            restaurant="",
            planet=None,
            source_document="",
            text=dish_name,
            normalized_text=normalize_text(dish_name),
            ingredients=set(),
            techniques=set(),
            licenses={},
        )
