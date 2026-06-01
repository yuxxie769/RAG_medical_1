from app.core.models import Document


def test_document_model_defaults():
    doc = Document(
        doc_id="doc_1",
        question="Q",
        answer="A",
        content="Q\nA",
    )

    assert doc.metadata == {}
    assert doc.doc_id == "doc_1"
    assert doc.question == "Q"
    assert doc.answer == "A"
    assert doc.content == "Q\nA"
