from __future__ import annotations
from typing import List, Dict, Any
from ..embeddings.provider import EmbeddingProvider
from .chunker import chunk_text


def ingest_document(text: str, provider: EmbeddingProvider, chunk_size: int = 200, overlap: int = 50) -> List[Dict[str, Any]]:
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    texts = [c["text"] for c in chunks]
    embs = provider.embed_batch(texts)
    out = []
    for c, e in zip(chunks, embs):
        out.append({
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "token_count": c["token_count"],
            "embedding": e,
        })
    return out
