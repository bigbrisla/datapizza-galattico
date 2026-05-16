from .models import QuerySpec
from .text_utils import tokenize


class QueryAnalyzer:
    """Lightweight query understanding for MVP retrieval orchestration."""

    def analyze(self, question: str) -> QuerySpec:
        return QuerySpec(original_text=question, tokens=tokenize(question))
