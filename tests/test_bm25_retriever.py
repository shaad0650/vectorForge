from vectorforge.core.retrieval.bm25 import BM25Retriever


def test_bm25_basic():
    docs = ["the quick brown fox", "jumps over the lazy dog", "the fox is quick and brown"]
    ids = ["d1", "d2", "d3"]
    r = BM25Retriever()
    r.build(docs, ids)
    res = r.search("quick fox", top_k=2)
    assert len(res) == 2
    assert res[0]["id"] in ids
