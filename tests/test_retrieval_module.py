from app.core.models import Document
from app.retrieval import (
    DummyEmbedder,
    HybridRetrievalRequest,
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
