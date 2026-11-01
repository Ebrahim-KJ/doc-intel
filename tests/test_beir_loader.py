import json

from src.evaluation import beir


def test_load_plain_layout(tmp_path, monkeypatch):
    d = tmp_path / "toy"
    (d / "qrels").mkdir(parents=True)
    (d / "corpus.jsonl").write_text("\n".join(json.dumps(x) for x in [
        {"_id": "d1", "title": "Aspirin", "text": "reduces fever"},
        {"_id": "d2", "title": "", "text": "unrelated"},
    ]))
    (d / "queries.jsonl").write_text("\n".join(json.dumps(x) for x in [
        {"_id": "q1", "text": "what lowers fever"},
        {"_id": "q2", "text": "no labels for me"},
    ]))
    (d / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    monkeypatch.setattr(beir, "DATA_DIR", tmp_path)

    corpus, queries, qrels = beir.load("toy")
    assert set(corpus) == {"d1", "d2"}
    assert queries == {"q1": "what lowers fever"}
    assert qrels == {"q1": {"d1": 1}}

    chunks = {c["chunk_id"]: c["text"] for c in beir.as_chunks(corpus, "toy")}
    assert chunks == {"d1": "Aspirin. reduces fever", "d2": "unrelated"}
