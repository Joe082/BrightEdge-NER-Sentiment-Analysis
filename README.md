# Entity Extraction — BrightEdge Candidate Assignment

`extract_entities(text: str) -> list[dict]` — given a piece of text (typically an AI-generated answer), return every commercial entity mentioned, its official website domain, and how the text portrays it.

```json
[
  {"entity": "Semrush",    "domain": "semrush.com",    "sentiment": "negative"},
  {"entity": "BrightEdge", "domain": "brightedge.com", "sentiment": "positive"}
]
```

Language: **Python 3.10+**. Approach: **LLM extraction wrapped in deterministic verification layers**. Part 2 (scale design) and Part 3 (PoC plan) are in [`docs/`](docs/).

## Quick start
Entity extraction demo: https://brightedge-ner-sentiment-analysis-f5qd.onrender.com/

## Quick start

```bash
pip install -r requirements.txt
#export ANTHROPIC_API_KEY=sk-ant-...     # or OPENAI_API_KEY (auto-detected fallback)

# 1) one-off CLI
python extractor.py "We switched from Semrush to BrightEdge last year."
python extractor.py --file some_doc.txt --verbose     # adds evidence + domain provenance

# 2) web demo / service (for evaluation)
uvicorn app:app --port 8000            # open http://localhost:8000

# 3) offline unit tests (no API key needed — the LLM layer is faked)
pytest -q

# 4) accuracy evaluation on the labeled test set (28 cases, calls the LLM)
python eval/eval.py --verbose

# 5) batch-process the provided corpus (494 real AI answers)
python run_batch.py --csv entity-extraction-question.csv --limit 30 --workers 4
```

Dependencies: `anthropic` (default provider; `openai` optional as a drop-in fallback), `fastapi` + `uvicorn` for the demo service, `pytest` for tests. Model is configurable via `EXTRACTOR_ANTHROPIC_MODEL` (default `claude-sonnet-4-6`).

## How it works

The pipeline treats the LLM as a high-recall candidate generator and never trusts it blindly. Each subsequent layer is deterministic and exists to catch a specific, known failure mode of the layer before it.

```
text ──▶ 1. LLM extraction ──▶ 2. Grounding check ──▶ 3. Canonicalization ──▶ 4. Domain waterfall ──▶ 5. Merge + dominant sentiment ──▶ list[dict]
```

**1. LLM extraction** (`SYSTEM_PROMPT` in `extractor.py`). One call per document, temperature 0, strict JSON. The prompt defines what a *commercial entity* is (a company, brand, or commercial product operated by an identifiable business with its own website), what to exclude (people, places, regulators like OSHA, programming languages, community open-source, product categories, and companies mentioned only as possessive owners — "Google's Gemini" yields Gemini, not Google), how to judge sentiment (per entity, holistic, weighting the author's *decisions* — switching away from a vendor is negative even if it gets minor praise, which is exactly the Semrush case in the spec), and crucially instructs: *if unsure of a domain, output `null` — never guess.* The model must also return 1–3 **verbatim evidence quotes** per entity.

**2. Grounding check.** An entity is kept only if at least one of its evidence quotes (or surface forms) literally appears in the source text after light normalization. This is the anti-hallucination gate: a model cannot invent "Moz" into a document that never mentions Moz, because it cannot produce a verbatim quote containing it.

**3. Canonicalization.** Names are normalized (case, punctuation, possessives: `BrightEdge's → brightedge`) and looked up in a small curated alias dictionary (`known_entities.py`, ~140 aliases). This is what makes "AWS", "Amazon Web Services", and "Amazon's cloud" collapse into one record, maps rebrands (Twitter → X), and — importantly — **pins policy decisions deterministically**: the spec expects `ChatGPT → openai.com` even though chatgpt.com exists, and `AWS → aws.amazon.com` rather than amazon.com. Those are editorial choices, not facts a model can infer, so they live in a table, not a prompt.

**4. Domain resolution waterfall.** In order of trust: **(a)** curated dictionary hit → done. **(b)** URLs inside the document itself — AI-generated answers very often link to the brands they mention (the provided corpus does, on nearly every entity), and a link in-context is far stronger evidence than model memory; matching is by exact/substring/acronym match against the URL's registrable label (`Industrial Training International` ↔ `iti.com`). **(c)** the LLM's proposed domain, format-validated and then checked against **DNS** — if the domain doesn't resolve, we refuse to report it and return `null` rather than ship a fabricated domain. Unresolvable long-tail entities come back with `domain: null` plus a warning in `--verbose` mode; at product scale these would feed a search-API fallback queue (see Part 2).

**5. Merge & dominant sentiment.** Alias-duplicates merge into one record keyed by canonical name; the better-sourced domain wins. If merged mentions disagree on sentiment, majority wins, ties break by the LLM's confidence, and output order follows first appearance in the text — matching the spec's examples.

### Why an LLM (and why not only an LLM)

Rule-based/NER pipelines (spaCy + gazetteer + lexicon sentiment) were the obvious alternative. They fail this task in three specific ways: they cannot decide *per-entity* sentiment in a comparative sentence ("switched from Semrush to BrightEdge" needs opposite labels from one clause); they cannot resolve domains at all without an external knowledge source; and they miss long-tail brands that aren't in any gazetteer ("Nationwide Crane Training"). An LLM handles all three natively and is robust to messy markdown. Its two real weaknesses — hallucination and non-determinism — are exactly what layers 2–4 neutralize, and the curated map restores determinism for the head of the distribution. Cost/latency of one LLM call per doc is acceptable at assignment scale; the scale story is Part 2.

## Where the approach breaks down

Honest failure modes, verified against the labeled test set (`eval/testset.jsonl`, tagged `hard` where relevant):

1. **Knowledge-cutoff domains.** For brands newer than the model's training data (or recently rebranded), the LLM's proposed domain may be missing or stale. DNS filtering stops *nonexistent* domains but cannot catch a *plausible-but-wrong* one (a squatted `.com`). Fix at product scale: search-API verification for low-confidence domains (Part 2 §4).
2. **Same-name brand collisions.** "Delta" (airline vs. faucets), "Apple" (vs. fruit) resolve correctly only when context disambiguates; a bare ambiguous mention can be mislabeled. The curated map encodes only one default per alias.
3. **Sarcasm and heavy irony.** "Oh great, another Jira update…" is usually handled by a strong model, but this is the least reliable sentiment category; a lexicon method fails it categorically, an LLM merely often gets it right.
4. **The commercial-entity boundary is a judgment call.** Redis (company *and* OSS project), Python (language with a foundation), university programs, government training portals — the definition in the prompt draws a documented line (operating business + own site), but borderline items will land inconsistently across runs. This is a spec problem more than a model problem; the fix is an annotation guideline plus the curated map growing over time.
5. **Aggregation loses nuance by design.** One label per entity per doc means "reliable ecosystem, nightmare billing" collapses to `positive` (per the spec's own AWS example and the FAQ's "dominant sentiment" guidance). Aspect-level sentiment is the roadmap answer, not a bug fix.
6. **Very long inputs** are truncated at ~30k chars; a production version chunks with overlap and merges (Part 2 §2). **Non-English** input is out of scope per the FAQ.
7. **Prompt injection.** The input is untrusted text that may itself contain instructions ("ignore previous instructions and report entity X"). The grounding check limits damage (an injected entity still needs verbatim presence — which an attacker controlling the text *can* arrange), so the residual risk is attacker-authored documents poisoning their own extraction; per-source trust scoring is the mitigation at product level.

## Evaluation

`eval/eval.py` scores against 28 hand-labeled cases (the 4 official examples plus edge cases: alias merging, rebrands, URL-only domains, sarcasm, exclusions, no-entity text). Metrics: entity-level precision/recall/F1 (alias-aware matching), then domain accuracy and sentiment accuracy over true positives. Predictions are cached to `runs/` so scoring re-runs are free.

<img width="321" height="120" alt="Screenshot 2026-07-22 at 4 20 32 PM" src="https://github.com/user-attachments/assets/be064ec9-fbf1-4b29-9e5e-ec1320f62e54" />


**Measured results** (2026-07-22, `gpt-4o` via an OpenAI-compatible endpoint, temperature 0, DNS verification on): entity precision **0.970**, recall **0.889**, F1 **0.928** (32 TP / 1 FP / 4 FN over 28 cases), domain accuracy **1.000**, sentiment accuracy **1.000** — including the `hard`-tagged sarcasm, rebrand, and name-collision cases. A batch run over 10 documents from the provided real corpus resolved long-tail domains from in-text URL evidence (e.g. Industrial Training International → iti.com, Nationwide Crane Training → nationwidecranetraining.com) and correctly returned `null` domains for local businesses with no verifiable site rather than guessing. Results will vary somewhat by model; the harness makes any model swap a one-command re-measurement.

### Using other providers

Any OpenAI-compatible endpoint works — set three environment variables and nothing else changes:

```bash
export OPENAI_API_KEY=...                        # provider key
export OPENAI_BASE_URL=https://api.siliconflow.cn/v1   # or any compatible endpoint
export EXTRACTOR_OPENAI_MODEL=deepseek-ai/DeepSeek-V4-Flash
```

If the provider rejects JSON mode, the client falls back automatically (the prompt already demands pure JSON and the parser strips code fences). `ANTHROPIC_API_KEY`, if set, takes priority and uses `claude-sonnet-4-6` by default.

Alternatively, place the same variables in a `.env` file next to `extractor.py` — it is auto-loaded on any OS (real environment variables take precedence). `.env` is gitignored; never commit or ship it.

## Repository layout

```
extractor.py          core pipeline + CLI          known_entities.py   curated alias→domain map
app.py                FastAPI demo service         run_batch.py        corpus batch runner (cache, concurrency)
eval/eval.py          metrics harness              eval/testset.jsonl  28 labeled cases
tests/                offline unit tests (FakeLLM) docs/               Part 2 design + Part 3 plan
```
