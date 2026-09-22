from vectorforge.core.ingestion.chunker import chunk_text
from vectorforge.core.embeddings.provider import EmbeddingProvider
from vectorforge.core.ingestion.pipeline import ingest_document


def test_chunker():
    text = "".join(["word "] * 500)
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 0


def test_embeddings_and_pipeline():
    provider = EmbeddingProvider()
    doc = "This is a test document about operating systems and scheduling." * 10
    items = ingest_document(doc, provider, chunk_size=20, overlap=5)
    assert len(items) > 0
    for it in items:
        assert "embedding" in it
