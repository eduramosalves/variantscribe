"""ColPali MaxSim late-interaction math + multi-vector index (no torch needed)."""

import numpy as np

from variantscribe.retrieval.colpali import MaxSimIndex, maxsim


def test_maxsim_identical_tokens():
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    # each query token finds an exact match -> 1 + 1 = 2
    assert maxsim(q, d) == 2.0


def test_maxsim_takes_best_doc_token_per_query_token():
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    d = np.array([[0.5, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert maxsim(q, d) == 1.0  # best match is the second doc token


def test_maxsim_empty_is_zero():
    assert maxsim(np.zeros((0, 2), dtype=np.float32), np.ones((3, 2), dtype=np.float32)) == 0.0


def test_index_search_ranks_by_maxsim():
    idx = MaxSimIndex()
    idx.add({"citation": "doc A p.1"}, np.array([[1.0, 0.0]], dtype=np.float32))
    idx.add({"citation": "doc B p.1"}, np.array([[0.0, 1.0]], dtype=np.float32))
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    hits = idx.search(q, k=2)
    assert hits[0][0]["citation"] == "doc A p.1"
    assert hits[0][1] > hits[1][1]


def test_index_save_load_roundtrip(tmp_path):
    idx = MaxSimIndex()
    idx.add({"citation": "x"}, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    idx.add({"citation": "y"}, np.array([[5.0, 6.0]], dtype=np.float32))
    path = tmp_path / "gl.npz"
    idx.save(path)

    loaded = MaxSimIndex.load(path)
    assert len(loaded) == 2
    q = np.array([[5.0, 6.0]], dtype=np.float32)
    assert loaded.search(q, k=1)[0][0]["citation"] == "y"
