import json
from unittest.mock import patch

from app.core.models import Document
from app.retrieval import (
    DummyEmbedder,
    HybridRetrievalRequest,
    LocalHTTPEmbedder,
    MilvusCollectionSchema,
    MilvusStore,
    RetrievalConfig,
    Retriever,
    VectorIndexer,
)


class DummyRetriever(Retriever):
    def retrieve(self, query: str, top_k: int | None = None):
        return []


class DummyStore(MilvusStore):
    def create_collection(self) -> None:
        return None

    def upsert(self, records):
        return None

    def search(self, query_vector, limit: int = 5):
        return []

    def delete_by_doc_ids(self, doc_ids):
        return None

    def count(self) -> int:
        return 0


class DummyIndexer(VectorIndexer):
    def index_documents(self, documents):
        return None

    def index_in_batches(self, documents):
        return None


def test_retriever_interface_can_be_instantiated():
    docs = [
        Document(
            doc_id="kb_001_000000",
            question="A",
            answer="1",
            content="Q: A\nA: 1",
        )
    ]

    retriever = DummyRetriever(documents=docs, config=RetrievalConfig(top_k=5))

    assert retriever.config.top_k == 5
    assert retriever.documents[0].doc_id == "kb_001_000000"


def test_embedding_and_milvus_skeleton_interfaces_exist():
    embedder = DummyEmbedder(dimension=4)
    vectors = embedder.embed_texts(["hello"])
    store = DummyStore(
        schema=MilvusCollectionSchema(collection_name="rag_medical_documents", dimension=4)
    )
    indexer = DummyIndexer(embedder=embedder, store=store, batch_size=2)

    assert len(vectors) == 1
    assert len(vectors[0]) == 4
    assert indexer.batch_size == 2
    assert store.schema.collection_name == "rag_medical_documents"
    assert HybridRetrievalRequest(query="abc", top_k=5).dense_weight == 0.5


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_local_http_embedder_reads_embeddings_from_json_endpoint():
    embedder = LocalHTTPEmbedder(
        endpoint="http://127.0.0.1:8001/embeddings",
        expected_dimension=3,
    )

    with patch(
        "app.retrieval.embedder.urllib_request.urlopen",
        return_value=_FakeHTTPResponse(
            {
                "count": 2,
                "dim": 3,
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            }
        ),
    ):
        vectors = embedder.embed_texts(["a", "b"])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_local_http_embedder_rejects_dimension_mismatch():
    embedder = LocalHTTPEmbedder(
        endpoint="http://127.0.0.1:8001/embeddings",
        expected_dimension=4,
    )

    with patch(
        "app.retrieval.embedder.urllib_request.urlopen",
        return_value=_FakeHTTPResponse(
            {
                "count": 1,
                "dim": 3,
                "embeddings": [[0.1, 0.2, 0.3]],
            }
        ),
    ):
        try:
            embedder.embed_texts(["a"])
            assert False, "expected dimension mismatch"
        except ValueError as exc:
            assert "Embedding dimension mismatch" in str(exc)
