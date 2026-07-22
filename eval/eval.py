"""Evaluation harness for extract_entities against a labeled test set.

Usage:
    export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY
    python eval/eval.py                              # full set
    python eval/eval.py --limit 5 --verbose          # quick smoke run
    python eval/eval.py --tags official              # only tagged cases
    python eval/eval.py --preds runs/preds.jsonl     # score saved predictions

Metrics
-------
- Entity identification: micro precision / recall / F1. Predicted and gold
  names are compared after canonicalization (alias map + normalization), so
  "Amazon Web Services" matches gold "AWS".
- Domain accuracy: among correctly identified entities, fraction whose domain
  equals gold (registrable-domain comparison, so www./scheme noise ignored).
- Sentiment accuracy: among correctly identified entities, fraction whose
  sentiment equals gold.

Each run also writes runs/preds-<timestamp>.jsonl so results are reproducible
and can be re-scored without paying for LLM calls again.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor import extract_entities, normalize_name, clean_domain  # noqa: E402
from known_entities import lookup  # noqa: E402


def canon_key(name: str) -> str:
    norm = normalize_name(name)
    hit = lookup(norm)
    return normalize_name(hit[0]) if hit else norm


def domains_equal(pred: str | None, gold: str | None) -> bool:
    p, g = clean_domain(pred), clean_domain(gold)
    if p is None or g is None:
        return p == g
    return p == g


def score_case(pred: list[dict], gold: list[dict]):
    gold_by_key = {canon_key(g["entity"]): g for g in gold}
    pred_by_key = {}
    for p in pred:
        pred_by_key.setdefault(canon_key(p["entity"]), p)  # first occurrence wins

    tp_keys = [k for k in pred_by_key if k in gold_by_key]
    fp_keys = [k for k in pred_by_key if k not in gold_by_key]
    fn_keys = [k for k in gold_by_key if k not in pred_by_key]

    domain_hits = sum(
        1 for k in tp_keys
        if domains_equal(pred_by_key[k].get("domain"), gold_by_key[k].get("domain"))
    )
    sent_hits = sum(
        1 for k in tp_keys
        if (pred_by_key[k].get("sentiment") or "").lower()
        == (gold_by_key[k].get("sentiment") or "").lower()
    )
    return {
        "tp": len(tp_keys), "fp": len(fp_keys), "fn": len(fn_keys),
        "domain_hits": domain_hits, "sent_hits": sent_hits,
        "fp_keys": fp_keys, "fn_keys": fn_keys,
        "tp_detail": [(k, pred_by_key[k], gold_by_key[k]) for k in tp_keys],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=str(ROOT / "eval" / "testset.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tags", default=None, help="comma-separated tag filter")
    ap.add_argument("--preds", default=None,
                    help="score an existing predictions jsonl instead of calling the LLM")
    ap.add_argument("--no-dns", action="store_true")
    ap.add_argument("--workers", type=int, default=6, help="parallel extraction workers")
    ap.add_argument("--verbose", action="store_true", help="print per-case diffs")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.testset, encoding="utf-8") if l.strip()]
    if args.tags:
        want = set(t.strip() for t in args.tags.split(","))
        cases = [c for c in cases if want & set(c.get("tags", []))]
    if args.limit:
        cases = cases[: args.limit]

    saved_preds = None
    if args.preds:
        saved_preds = {p["id"]: p["pred"] for p in
                       (json.loads(l) for l in open(args.preds, encoding="utf-8") if l.strip())}

    runs_dir = ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)
    out_path = runs_dir / f"preds-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    totals = {"tp": 0, "fp": 0, "fn": 0, "domain_hits": 0, "sent_hits": 0}
    t0 = time.time()

    # Run extraction up-front (optionally in parallel), then score in order.
    all_preds: dict = {}
    if saved_preds is None:
        from concurrent.futures import ThreadPoolExecutor

        def _run(case):
            try:
                return case["id"], extract_entities(case["text"], verify_dns=not args.no_dns)
            except Exception as e:
                print(f"[case {case['id']}] extraction error: {e}")
                return case["id"], []

        workers = max(1, args.workers)
        if workers == 1:
            for c in cases:
                cid, p = _run(c)
                all_preds[cid] = p
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for cid, p in ex.map(_run, cases):
                    all_preds[cid] = p

    with open(out_path, "w", encoding="utf-8") as fout:
        for case in cases:
            if saved_preds is not None:
                pred = saved_preds.get(case["id"], [])
            else:
                pred = all_preds.get(case["id"], [])
            fout.write(json.dumps({"id": case["id"], "pred": pred}, ensure_ascii=False) + "\n")

            s = score_case(pred, case["expected"])
            for k in ("tp", "fp", "fn", "domain_hits", "sent_hits"):
                totals[k] += s[k]

            wrong_domain = [k for k, p, g in s["tp_detail"]
                            if not domains_equal(p.get("domain"), g.get("domain"))]
            wrong_sent = [k for k, p, g in s["tp_detail"]
                          if (p.get("sentiment") or "").lower() != (g.get("sentiment") or "").lower()]
            imperfect = s["fp_keys"] or s["fn_keys"] or wrong_domain or wrong_sent
            if args.verbose or imperfect:
                mark = "OK " if not imperfect else "DIFF"
                print(f"[{mark}] case {case['id']:>3} tags={','.join(case.get('tags', []))}")
                if imperfect:
                    if s["fp_keys"]:
                        print(f"        spurious entities : {s['fp_keys']}")
                    if s["fn_keys"]:
                        print(f"        missed entities   : {s['fn_keys']}")
                    for k, p, g in s["tp_detail"]:
                        if k in wrong_domain:
                            print(f"        domain[{k}]  pred={p.get('domain')!r} gold={g.get('domain')!r}")
                        if k in wrong_sent:
                            print(f"        sentiment[{k}] pred={p.get('sentiment')!r} gold={g.get('sentiment')!r}")

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    dom = totals["domain_hits"] / tp if tp else 0.0
    sent = totals["sent_hits"] / tp if tp else 0.0

    print("\n================ SUMMARY ================")
    print(f"cases                 : {len(cases)}")
    print(f"entity precision      : {prec:.3f}   (TP={tp} FP={fp})")
    print(f"entity recall         : {rec:.3f}   (FN={fn})")
    print(f"entity F1             : {f1:.3f}")
    print(f"domain accuracy  (TP) : {dom:.3f}")
    print(f"sentiment accuracy(TP): {sent:.3f}")
    print(f"elapsed               : {time.time() - t0:.1f}s")
    print(f"predictions saved     : {out_path}")


if __name__ == "__main__":
    main()
