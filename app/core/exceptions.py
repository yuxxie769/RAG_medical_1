class RAGError(Exception):
    """Base error for the RAG project."""


class DataValidationError(RAGError):
    """Raised when raw data is invalid."""


class RetrievalError(RAGError):
    """Raised when retrieval fails."""


class GenerationError(RAGError):
    """Raised when generation fails."""
