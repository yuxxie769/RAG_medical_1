import json

from app.core.models import RetrievalResult
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rerank import NoopReranker, OpenAICompatibleReranker, RerankResult


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [[0.1, 0.2] for _ in texts]


class FakeStore:
    def __init__(self, results):
        self.results = results
        self.search_calls = []

    def search(self, query_vector=None, query_text=None, limit=5, search_method="hybrid"):
        self.search_calls.append(
            {
                "query_vector": query_vector,
                "query_text": query_text,
                "limit": limit,
                "search_method": search_method,
            }
        )
        return list(self.results)


class StaticReranker:
    provider_name = "openai_compatible"

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def rerank(self, query, results):
        self.calls.append((query, [result.doc_id for result in results]))
        return [RerankResult(doc_id=result.doc_id, score=self.scores[result.doc_id]) for result in results]


class FailingReranker:
    provider_name = "openai_compatible"

    def rerank(self, query, results):
        raise RuntimeError("rerank quota exceeded")


def make_results():
    return [
        RetrievalResult(doc_id="doc_1", content="A", score=0.9, metadata={"source": "dense"}),
        RetrievalResult(doc_id="doc_2", content="B", score=0.8, metadata={"source": "dense"}),
        RetrievalResult(doc_id="doc_3", content="C", score=0.3, metadata={"source": "dense"}),
    ]


def test_noop_rerank_preserves_original_order_and_adds_metadata():
    retriever = HybridRetriever(
        embedder=FakeEmbedder(),
        store=FakeStore(make_results()),
        reranker=NoopReranker(),
        rerank_candidate_limit=15,
    )

    results = retriever.retrieve(query="Q", limit=3, search_method="dense", min_score=0.5, top_k=2)

    assert [result.doc_id for result in results] == ["doc_1", "doc_2"]
    assert results[0].score == 0.9
    assert results[0].metadata["retrieval_score"] == 0.9
    assert results[0].metadata["rerank_score"] is None
    assert results[0].metadata["rerank_applied"] is False
    assert results[0].metadata["rerank_provider"] == "none"


def test_successful_rerank_reorders_candidates_after_min_score_filter():
    retriever = HybridRetriever(
        embedder=FakeEmbedder(),
        store=FakeStore(make_results()),
        reranker=StaticReranker({"doc_1": 0.2, "doc_2": 0.95}),
        rerank_candidate_limit=2,
    )

    results = retriever.retrieve(query="Q", limit=3, search_method="hybrid", min_score=0.5, top_k=3)

    assert [result.doc_id for result in results] == ["doc_2", "doc_1"]
    assert results[0].score == 0.95
    assert results[0].metadata["retrieval_score"] == 0.8
    assert results[0].metadata["rerank_score"] == 0.95
    assert results[0].metadata["rerank_applied"] is True


def test_rerank_only_applies_to_candidate_limit_and_appends_remaining_results():
    retriever = HybridRetriever(
        embedder=FakeEmbedder(),
        store=FakeStore(
            [
                RetrievalResult(doc_id="doc_1", content="A", score=0.9, metadata={}),
                RetrievalResult(doc_id="doc_2", content="B", score=0.8, metadata={}),
                RetrievalResult(doc_id="doc_3", content="C", score=0.7, metadata={}),
            ]
        ),
        reranker=StaticReranker({"doc_1": 0.1, "doc_2": 0.95}),
        rerank_candidate_limit=2,
    )

    results = retriever.retrieve(query="Q", limit=3, search_method="hybrid", min_score=0.0, top_k=3)

    assert [result.doc_id for result in results] == ["doc_2", "doc_1", "doc_3"]
    assert results[2].score == 0.7
    assert results[2].metadata["rerank_applied"] is False
    assert results[2].metadata["rerank_provider"] == "openai_compatible"


def test_rerank_failure_falls_back_to_original_order():
    retriever = HybridRetriever(
        embedder=FakeEmbedder(),
        store=FakeStore(make_results()),
        reranker=FailingReranker(),
        rerank_candidate_limit=2,
    )

    results = retriever.retrieve(query="Q", limit=3, search_method="hybrid", min_score=0.5, top_k=2)

    assert [result.doc_id for result in results] == ["doc_1", "doc_2"]
    assert results[0].metadata["rerank_applied"] is False
    assert results[0].metadata["rerank_provider"] == "openai_compatible"


def test_openai_compatible_reranker_parses_cohere_style_response(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.91},
                        {"index": 0, "relevance_score": 0.52},
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert request.full_url == "https://example.com/rerank"
        assert timeout == 12
        body = json.loads(request.data.decode("utf-8"))
        assert body["model"] == "rerank-v1"
        assert body["query"] == "Q"
        assert body["documents"] == ["A", "B"]
        assert body["top_n"] == 2
        return FakeResponse()

    monkeypatch.setattr("app.retrieval.rerank.urllib_request.urlopen", fake_urlopen)
    reranker = OpenAICompatibleReranker(
        api_key="key",
        base_url="https://example.com",
        path="/rerank",
        model="rerank-v1",
        timeout=12,
    )
    results = reranker.rerank(
        "Q",
        [
            RetrievalResult(doc_id="doc_1", content="A", score=0.7, metadata={}),
            RetrievalResult(doc_id="doc_2", content="B", score=0.6, metadata={}),
        ],
    )

    assert [(result.doc_id, result.score) for result in results] == [("doc_2", 0.91), ("doc_1", 0.52)]
