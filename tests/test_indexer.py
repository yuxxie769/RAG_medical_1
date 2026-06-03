from app.core.models import Document
from app.retrieval import InMemoryMilvusStore, MilvusCollectionSchema, VectorIndexer


class ShortEmbedder:
    def embed_texts(self, texts):
        return []


class SuccessfulEmbedder:
    def embed_texts(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FailingMilvusStore:
    def create_collection(self):
        raise RuntimeError("Milvus unavailable")


def test_indexer_reports_embedding_count_mismatch():
    store = InMemoryMilvusStore(MilvusCollectionSchema(collection_name="test", dimension=2))
    indexer = VectorIndexer(embedder=ShortEmbedder(), store=store)
    document = Document(doc_id="doc_1", question="Q", answer="A", content="Q: Q\nA: A")

    result = indexer.index_documents([document])

    assert result.failed_count == 1
    assert result.failed_batches == 1
    assert result.milvus_failed_batches == 0
    assert result.failure_stage == "embedding"
    assert "Embedding count mismatch" in result.errors[0]


def test_indexer_classifies_milvus_write_failure():
    indexer = VectorIndexer(embedder=SuccessfulEmbedder(), store=FailingMilvusStore())
    document = Document(doc_id="doc_1", question="Q", answer="A", content="Q: Q\nA: A")

    result = indexer.index_documents([document])

    assert result.failed_count == 1
    assert result.failed_batches == 1
    assert result.milvus_failed_batches == 1
    assert result.failure_stage == "milvus"
    assert result.errors == ["Milvus unavailable"]
