"""Load BEIR-format retrieval datasets (corpus, queries, graded qrels)."""

from __future__ import annotations

import csv
import gzip
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Tuple

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "beir"
OFFICIAL_ZIP = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
ZIP_MIRROR = "https://github.com/liyongkang123/extended_beir_datasets/releases/download/beir_v1.0/{name}.zip"


def _find(folder: Path, stem: str) -> Path:
    for cand in (folder / stem, folder / f"{stem}.gz", folder / "qrels" / stem):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{stem} not found in {folder}")


def _read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def fetch(name: str) -> Path:
    """Download a BEIR dataset zip (official host first, then a GitHub release mirror)."""
    import requests

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    last = None
    for url in (OFFICIAL_ZIP, ZIP_MIRROR):
        try:
            r = requests.get(url.format(name=name), timeout=300)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(DATA_DIR)
            return DATA_DIR / name
        except Exception as exc:
            last = exc
    raise RuntimeError(f"could not download {name}: {last}")


def load(name: str) -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    folder = DATA_DIR / name
    if not folder.exists():
        folder = fetch(name)
    corpus = {str(d["_id"]): {"title": d.get("title", "") or "", "text": d.get("text", "") or ""}
              for d in _read_jsonl(_find(folder, "corpus.jsonl"))}
    qrels: Dict[str, Dict[str, int]] = {}
    with open(_find(folder, "test.tsv"), encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    queries = {str(q["_id"]): q["text"] for q in _read_jsonl(_find(folder, "queries.jsonl")) if str(q["_id"]) in qrels}
    qrels = {q: r for q, r in qrels.items() if q in queries}
    return corpus, queries, qrels


def as_chunks(corpus: Dict[str, dict], source: str) -> list[dict]:
    """One chunk per document, in the same shape the ingestion pipeline produces."""
    return [{"text": f"{d['title']}. {d['text']}" if d["title"] else d["text"],
             "source": source, "page": 0, "doc_id": doc_id, "chunk_id": doc_id, "chunk_idx": 0}
            for doc_id, d in corpus.items()]
