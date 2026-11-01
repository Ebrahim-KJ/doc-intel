"""Load BEIR-format retrieval datasets (corpus, queries, graded qrels)."""

from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path
from typing import Dict, Tuple

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "beir"


def _read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def fetch(name: str) -> Path:
    """Download a BEIR dataset from the Hugging Face hub into data/beir/<name>."""
    from huggingface_hub import hf_hub_download

    dest = DATA_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ("corpus.jsonl.gz", "queries.jsonl.gz"):
        if not (dest / fname).exists():
            p = hf_hub_download(f"BeIR/{name}", fname, repo_type="dataset", local_dir=dest)
            os.replace(p, dest / fname) if Path(p) != dest / fname else None
    if not (dest / "test.tsv").exists():
        p = hf_hub_download(f"BeIR/{name}-qrels", "test.tsv", repo_type="dataset", local_dir=dest)
        os.replace(p, dest / "test.tsv") if Path(p) != dest / "test.tsv" else None
    return dest


def load(name: str) -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    folder = DATA_DIR / name
    if not folder.exists():
        folder = fetch(name)
    corpus = {d["_id"]: {"title": d.get("title", ""), "text": d.get("text", "")}
              for d in _read_jsonl(folder / "corpus.jsonl.gz")}
    qrels: Dict[str, Dict[str, int]] = {}
    with open(folder / "test.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    queries = {q["_id"]: q["text"] for q in _read_jsonl(folder / "queries.jsonl.gz") if q["_id"] in qrels}
    return corpus, queries, qrels


def as_chunks(corpus: Dict[str, dict], source: str) -> list[dict]:
    """One chunk per document, in the same shape the ingestion pipeline produces."""
    return [{"text": (d["title"] + ". " + d["text"]).strip(". ") if d["title"] else d["text"],
             "source": source, "page": 0, "doc_id": doc_id, "chunk_id": doc_id, "chunk_idx": 0}
            for doc_id, d in corpus.items()]
