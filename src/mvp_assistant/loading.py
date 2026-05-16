from pathlib import Path

from .document_loader import _read_pdf
from .models import SourceDocument


class DocumentLoader:
    """Loading step: read source menu documents from disk."""

    def load_menu_documents(self, repo_root: Path) -> list[SourceDocument]:
        menu_dir = repo_root / "Dataset" / "knowledge_base" / "menu"
        documents: list[SourceDocument] = []
        for pdf_path in sorted(menu_dir.glob("*.pdf")):
            documents.append(SourceDocument(source_document=pdf_path.name, text=_read_pdf(pdf_path)))
        return documents
