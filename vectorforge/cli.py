def main():
    import sys
    from pathlib import Path
    if len(sys.argv) < 2:
        print("VectorForge CLI. Available commands: build-index")
        return
    cmd = sys.argv[1]
    if cmd == "build-index":
        # usage: vectorforge build-index embeddings.npz out.index
        if len(sys.argv) < 4:
            print("Usage: vectorforge build-index embeddings.npz out.index")
            return
        inp = sys.argv[2]
        out = sys.argv[3]
        from vectorforge.core.indexing.hnsw_builder import build_hnsw_index_from_npz
        build_hnsw_index_from_npz(inp, out)
        print("Index written to", out)
    elif cmd == "ingest":
        # usage: vectorforge ingest <input_path> <out.npz> [collection_name]
        if len(sys.argv) < 4:
            print("Usage: vectorforge ingest <input_path> <out.npz> [collection_name]")
            return
        inp = sys.argv[2]
        out = sys.argv[3]
        collection_name = sys.argv[4] if len(sys.argv) >= 5 else None
        from pathlib import Path
        from vectorforge.core.embeddings.provider import EmbeddingProvider
        from vectorforge.core.ingestion.pipeline import ingest_document
        import json
        import os

        p = Path(inp)
        provider = EmbeddingProvider()
        all_vecs = []
        all_ids = []
        all_texts = []
        if p.is_dir():
            for fp in sorted(p.glob("*.txt")):
                text = fp.read_text(encoding="utf-8")
                items = ingest_document(text, provider)
                for it in items:
                    all_vecs.append(it["embedding"])
                    cid = f"{fp.stem}_chunk{it['chunk_index']}"
                    all_ids.append(cid)
                    all_texts.append(it["text"])
        else:
            text = p.read_text(encoding="utf-8")
            items = ingest_document(text, provider)
            for it in items:
                all_vecs.append(it["embedding"])
                cid = f"doc_chunk{it['chunk_index']}"
                all_ids.append(cid)
                all_texts.append(it["text"])

        import numpy as np
        vecs = np.array(all_vecs, dtype=np.float32)
        ids = np.array(all_ids, dtype=object)
        texts = np.array(all_texts, dtype=object)

        # prepare data directory
        base = Path("data") / "collections"
        os.makedirs(base, exist_ok=True)

        # create collection metadata if name provided
        import uuid
        collection_id = None
        metadata = None
        if collection_name:
            collection_id = str(uuid.uuid4())
            metadata = {
                "id": collection_id,
                "name": collection_name,
                "embedding_model": provider.model_name,
                "embedding_dimension": provider.dimension(),
                "distance_metric": "cosine",
            }
            # save metadata file
            meta_path = base / f"{collection_id}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            # update registry
            registry_path = base / "collections.json"
            try:
                if registry_path.exists():
                    with open(registry_path, "r", encoding="utf-8") as f:
                        registry = json.load(f)
                else:
                    registry = {}
            except Exception:
                registry = {}
            registry[collection_id] = {"name": collection_name, "embedding_dimension": provider.dimension()}
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)

            # default out path to per-collection if user passed placeholder
            out_path = base / f"{collection_id}.npz"
        else:
            out_path = Path(out)

        # save embeddings + texts + ids
        np.savez_compressed(out_path, vectors=vecs, ids=ids, texts=texts)
        print(f"Ingested {len(all_ids)} chunks, wrote {out_path}")
    else:
        print(f"Unknown command: {cmd}")
