"""Batch-process a corpus of documents (e.g. the provided
`entity-extraction-question.csv`, one column `input_doc`).

Usage:
    export ANTHROPIC_API_KEY=...
    python run_batch.py --csv entity-extraction-question.csv --limit 20
    python run_batch.py --csv corpus.csv --workers 4 --out runs/batch.jsonl

Features:
- content-hash cache in .cache/ so re-runs never pay twice for the same doc
- bounded thread pool (LLM calls are I/O bound) with per-doc error isolation
- writes one JSON line per doc: {"doc_id", "sha", "entities": [...]}
- prints a corpus summary: top entities by mention count and their sentiment mix
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extractor import extract_entities, default_client  # noqa: E402

CACHE_DIR = Path(".cache")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_docs(args) -> list[str]:
    if args.csv:
        with open(args.csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        col = args.column or ("input_doc" if "input_doc" in rows[0] else list(rows[0])[0])
        return [r[col] for r in rows if r.get(col, "").strip()]
    if args.jsonl:
        return [json.loads(l)["text"] for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    raise SystemExit("provide --csv or --jsonl")


def process_one(i: int, text: str, verify_dns: bool, verbose: bool) -> dict:
    key = sha(text)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    try:
        entities = extract_entities(text, verify_dns=verify_dns, verbose=verbose)
        rec = {"doc_id": i, "sha": key, "entities": entities}
    except Exception as e:
        rec = {"doc_id": i, "sha": key, "entities": [], "error": str(e)}
    cache_file.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--jsonl")
    ap.add_argument("--column", help="CSV column holding the document text")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/batch_results.jsonl")
    ap.add_argument("--no-dns", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="keep evidence / provenance fields in the output")
    args = ap.parse_args()

    default_client()  # fail fast if no key configured
    docs = load_docs(args)
    if args.limit:
        docs = docs[: args.limit]
    CACHE_DIR.mkdir(exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"processing {len(docs)} docs with {args.workers} workers ...")
    t0 = time.time()
    results: list[dict] = [None] * len(docs)  # type: ignore
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, i, d, not args.no_dns, args.verbose): i
                for i, d in enumerate(docs)}
        done = 0
        for fut in cf.as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if done % 10 == 0 or done == len(docs):
                print(f"  {done}/{len(docs)}  ({time.time()-t0:.0f}s)")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- corpus summary ----
    mention_counts: Counter[str] = Counter()
    sentiments: dict[str, Counter] = defaultdict(Counter)
    domains: dict[str, str] = {}
    errors = 0
    for r in results:
        if r.get("error"):
            errors += 1
        for e in r["entities"]:
            mention_counts[e["entity"]] += 1
            sentiments[e["entity"]][e["sentiment"]] += 1
            if e.get("domain"):
                domains[e["entity"]] = e["domain"]

    print(f"\ndone in {time.time()-t0:.0f}s, errors={errors}, output={args.out}")
    print(f"{'entity':<38}{'domain':<32}{'docs':>5}  pos/neu/neg")
    for name, n in mention_counts.most_common(20):
        s = sentiments[name]
        print(f"{name:<38}{domains.get(name, '-'):<32}{n:>5}  "
              f"{s['positive']}/{s['neutral']}/{s['negative']}")


if __name__ == "__main__":
    main()
