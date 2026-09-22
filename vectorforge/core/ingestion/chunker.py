from __future__ import annotations
from typing import List, Dict, Any

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[Dict[str, Any]]:
    words = text.split()
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({"chunk_index": idx, "text": chunk_text, "token_count": len(chunk_words)})
        idx += 1
        i += chunk_size - overlap
    return chunks
