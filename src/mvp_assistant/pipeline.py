from pathlib import Path

from .answer_generation import CsvAnswerWriter
from .document_loader import load_knowledge_base_corpus
from .index_builder import DishIndexBuilder
from .io_utils import load_mapping, load_question_items, load_questions, write_submission
from .knowledge_config import KnowledgeConfig
from .llm_client import build_llm_client
from .loading import DocumentLoader
from .metadata_extraction import RecipeMetadataExtractor
from .nosql_store import RecipeMetadataStore
from .parsing import MenuParser
from .query_analyzer import QueryAnalyzer
from .query_transform import MetadataQueryTransformer
from .retriever import LexicalRetriever
from .semantic_chunking import SemanticMenuChunker


class MvpAssistantPipeline:
    def __init__(self) -> None:
        self.query_analyzer = QueryAnalyzer()
        self.index_builder = DishIndexBuilder()
        self.retriever = LexicalRetriever()
        self.loader = DocumentLoader()
        self.parser = MenuParser()
        self.chunker = SemanticMenuChunker()
        self.llm_client = build_llm_client()
        self.metadata_extractor = RecipeMetadataExtractor(self.llm_client)
        self.answer_writer = CsvAnswerWriter()

    def run(self, repo_root: Path, questions_path: Path, mapping_path: Path, output_path: Path) -> Path:
        question_items = load_question_items(questions_path)
        questions = [item["domanda"] for item in question_items]
        difficulty_by_row = {idx: item.get("difficoltà", "") for idx, item in enumerate(question_items, start=1)}
        store = self._build_metadata_store(repo_root, mapping_path)
        query_transformer = MetadataQueryTransformer(store.all(), questions, self.llm_client, difficulty_by_row)

        rows: list[tuple[int, list[int]]] = []
        for idx, question in enumerate(questions, start=1):
            query = query_transformer.transform(question, row_id=idx)
            dish_ids = store.find(query)
            rows.append((idx, dish_ids))

        return self.answer_writer.write(output_path, rows)

    def _build_metadata_store(self, repo_root: Path, mapping_path: Path) -> RecipeMetadataStore:
        dish_mapping = load_mapping(mapping_path)
        self.parser = MenuParser(KnowledgeConfig(repo_root).planet_names())
        documents = self.loader.load_menu_documents(repo_root)
        seen_ids: set[int] = set()
        recipes = []

        for document in documents:
            parsed_menu = self.parser.parse(document)
            chunks = self.chunker.chunk(parsed_menu, dish_mapping, seen_ids)
            recipes.extend(self.metadata_extractor.extract(chunk) for chunk in chunks)

        for dish_name, dish_id in dish_mapping.items():
            if dish_id not in seen_ids:
                recipes.append(self.metadata_extractor.placeholder(dish_name, dish_id))

        store = RecipeMetadataStore(self.llm_client)
        store.insert_many(recipes)
        return store

    def run_lexical(self, repo_root: Path, questions_path: Path, mapping_path: Path, output_path: Path) -> Path:
        kb_dir = repo_root / "Dataset" / "knowledge_base"
        dish_mapping = load_mapping(mapping_path)
        corpus = load_knowledge_base_corpus(kb_dir)
        docs = self.index_builder.build(corpus=corpus, dish_mapping=dish_mapping)

        rows: list[tuple[int, list[int]]] = []
        for idx, question in enumerate(load_questions(questions_path), start=1):
            query = self.query_analyzer.analyze(question)
            rows.append((idx, self.retriever.retrieve(query, docs)))

        write_submission(output_path, rows)
        return output_path
