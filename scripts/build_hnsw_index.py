"""Build an HNSW index from embeddings saved in a .npz file.

Usage:
    python scripts/build_hnsw_index.py data.npz out.index
"""
import sys
from vectorforge.core.indexing.hnsw_builder import build_hnsw_index_from_npz


def main():
    if len(sys.argv) < 3:
        print("Usage: build_hnsw_index.py embeddings.npz out.index")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2]
    build_hnsw_index_from_npz(inp, out)
    print("Saved index to", out)


if __name__ == "__main__":
    main()
