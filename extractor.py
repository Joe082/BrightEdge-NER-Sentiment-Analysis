"""
Entity extraction for brand monitoring.

    extract_entities(text: str) -> list[dict]
    # -> [{"entity": ..., "domain": ..., "sentiment": ...}, ...]

Pipeline (each layer catches a different failure mode of the previous one):

    1. LLM extraction        - candidate entities + evidence quotes + sentiment
                               + suggested domain ("null if unsure").
    2. Grounding validation  - every candidate must be literally present in the
                               source text (via its evidence quotes). Kills
                               hallucinated entities.
    3. Canonicalization      - normalize surface forms; merge aliases via a
                               curated dictionary (AWS == "Amazon Web Services"
                               == "Amazon's cloud").
    4. Domain resolution     - waterfall: curated map > URLs found in the text
                               > LLM knowledge, then optional DNS existence
                               check. Kills hallucinated domains.
    5. Merge & dominant      - duplicates collapse to one record; conflicting
       sentiment               sentiments resolve by majority, then confidence.

Run `python extractor.py --help` for CLI usage.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import socket
import sys
import time
import unicodedata
from dataclasses import dataclass, field

from known_entities import lookup

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def _load_dotenv() -> None:
    """Minimal, dependency-free .env loader.

    Reads KEY=VALUE lines from a `.env` file next to this script so the tool
    runs identically on Windows/macOS/Linux without shell-specific `export`
    steps. Supports '#' comments, an optional leading 'export ', and quoted
    values. Real environment variables always take precedence. Keep `.env`
    out of version control (it is gitignored).
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:  # -sig: tolerate Windows BOM
            lines = f.read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, val = line.partition("=")
        if sep:
            key = key.strip()
            val = val.strip().strip("'\"")
            if key:
                os.environ.setdefault(key, val)


_load_dotenv()

VALID_SENTIMENTS = {"positive", "negative", "neutral"}
MAX_INPUT_CHARS = 30_000          # ~7.5k tokens; longer inputs are truncated with a warning
DNS_TIMEOUT_SECONDS = 2.0
LLM_MAX_RETRIES = 2

DEFAULT_ANTHROPIC_MODEL = os.environ.get("EXTRACTOR_ANTHROPIC_MODEL", "claude-sonnet-4-6")
DEFAULT_OPENAI_MODEL = os.environ.get("EXTRACTOR_OPENAI_MODEL", "gpt-4o")

SYSTEM_PROMPT = """You are an information-extraction engine for a brand-monitoring product. \
Given one English text (often an AI chatbot answer), extract every COMMERCIAL ENTITY it mentions.

DEFINITION - a commercial entity is a company, brand, or commercial product/service that is \
operated by an identifiable business and has (or plausibly has) its own official website. \
Examples: Salesforce, AWS, ChatGPT, a local training company, a hotel chain.

EXCLUDE:
- People, places, job titles, events.
- Generic technologies, programming languages, protocols, standards, and community open-source \
projects with no operating company behind them (e.g. "SQL", "Python", "HTTP", "PostgreSQL").
- Government bodies, regulators and standards organizations (e.g. "OSHA", "ISO") - they are \
referenced as authorities, not as brands being portrayed.
- Product categories and generic nouns ("a CRM", "cloud providers").
- A company mentioned ONLY as a possessive owner of a product ("Google's Gemini" -> extract \
Gemini, not Google), unless the company itself is independently discussed.

NORMALIZATION: report each entity once, under its canonical, currently-official name \
(e.g. "Amazon Web Services" and "AWS" -> "AWS"; "Twitter" -> "X"). Keep original casing style \
of the official brand.

SENTIMENT - judge how THIS text portrays THIS entity (not the brand's general reputation), \
one holistic label per entity:
- "positive": praised, recommended, chosen, described as reliable/better/the backbone, or the \
author switched TO it.
- "negative": criticized, described as worse/expensive/frustrating, rejected, or the author \
switched AWAY from it - even if it also gets minor praise.
- "neutral": listed or described factually with no clear stance.
If sentiment is mixed, pick the DOMINANT one: weight the author's decisions and conclusions \
(what they chose, recommend, or abandoned) over incidental remarks. Never invent a stance that \
is not in the text.

DOMAIN: the entity's primary official website domain, lowercase, no scheme/www/path \
(e.g. "salesforce.com", "aws.amazon.com"). A product without its own standalone primary site \
resolves to its operating company's domain (ChatGPT -> "openai.com"). If the text itself links \
to the entity's official site, prefer that URL's domain. If you are NOT confident, output null - \
never guess or fabricate a domain.

EVIDENCE: for every entity give 1-3 short VERBATIM quotes copied exactly from the text \
(substrings, including the entity mention). These are used to verify you did not invent the entity.

OUTPUT: ONLY a valid JSON object, no markdown fences, no commentary:
{"entities": [
  {"name": "<canonical name>",
   "surface_forms": ["<as it appears in text>", ...],
   "evidence": ["<verbatim quote>", ...],
   "sentiment": "positive|negative|neutral",
   "sentiment_reason": "<one short sentence>",
   "domain": "<domain or null>",
   "is_commercial": true|false,
   "confidence": <0.0-1.0>}
]}
If there are no commercial entities, output {"entities": []}."""


# --------------------------------------------------------------------------
# LLM clients (thin, swappable). Anthropic preferred, OpenAI fallback,
# FakeLLM for offline tests.
# --------------------------------------------------------------------------

class LLMError(RuntimeError):
    pass


class AnthropicClient:
    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL):
        from anthropic import Anthropic  # lazy import
        self._client = Anthropic()       # reads ANTHROPIC_API_KEY
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=3000,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class OpenAIClient:
    """Works with the official OpenAI API and any OpenAI-compatible endpoint
    (SiliconFlow, DeepSeek, Together, vLLM, ...). Set OPENAI_BASE_URL to point
    at a compatible provider, e.g. https://api.siliconflow.cn/v1
    """

    def __init__(self, model: str = DEFAULT_OPENAI_MODEL,
                 base_url: str | None = None):
        from openai import OpenAI        # lazy import
        base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self._client = OpenAI(base_url=base_url)  # reads OPENAI_API_KEY
        self.model = model
        self._json_mode_ok = True  # some providers reject response_format

    def complete(self, system: str, user: str) -> str:
        kwargs = dict(
            model=self.model,
            temperature=0,
            max_tokens=3000,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if self._json_mode_ok:
            try:
                resp = self._client.chat.completions.create(
                    response_format={"type": "json_object"}, **kwargs)
                return resp.choices[0].message.content or ""
            except Exception:
                # Provider likely doesn't support json mode; fall back once
                # and remember (the prompt itself already demands pure JSON).
                self._json_mode_ok = False
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


def default_client():
    """Pick a provider based on which API key is set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIClient()
    raise LLMError(
        "No LLM credentials found. Set ANTHROPIC_API_KEY (preferred) or OPENAI_API_KEY "
        "(plus optional OPENAI_BASE_URL / EXTRACTOR_OPENAI_MODEL for OpenAI-compatible providers)."
    )


# --------------------------------------------------------------------------
# Text / name normalization helpers
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s.-]", re.UNICODE)


def normalize_name(name: str) -> str:
    """Normalize a brand name for dictionary lookup and dedup keys."""
    s = unicodedata.normalize("NFKC", name or "")
    s = s.replace("\u2019", "'").lower().strip()
    s = re.sub(r"'s\b", "", s)            # possessive: "brightedge's" -> "brightedge"
    s = s.replace("'", "")
    s = _PUNCT_RE.sub(" ", s)
    s = s.replace(".", " ").replace("-", " ")
    return _WS_RE.sub(" ", s).strip()


def _normalize_for_matching(text: str) -> str:
    """Loose normalization of the source text for evidence grounding."""
    s = unicodedata.normalize("NFKC", text)
    s = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("*", "").replace("_", "")   # markdown emphasis
    return _WS_RE.sub(" ", s).lower()


def is_grounded(candidate: dict, norm_text: str) -> bool:
    """An entity is grounded iff at least one evidence quote (or a surface form,
    or the name itself) literally appears in the text."""
    probes: list[str] = []
    probes += [e for e in candidate.get("evidence", []) if isinstance(e, str)]
    probes += [s for s in candidate.get("surface_forms", []) if isinstance(s, str)]
    if isinstance(candidate.get("name"), str):
        probes.append(candidate["name"])
    for p in probes:
        p_norm = _WS_RE.sub(" ", p.replace("*", "").replace("_", "")).strip().lower()
        if len(p_norm) >= 2 and p_norm in norm_text:
            return True
    return False


# --------------------------------------------------------------------------
# Domain handling
# --------------------------------------------------------------------------

_DOMAIN_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}$")
_URL_RE = re.compile(r"https?://([^\s/\)\]\">,]+)", re.IGNORECASE)

# Domains that appear in AI-answer boilerplate but are never the entity's own site.
_IGNORED_URL_DOMAINS = {
    "chatgpt.com", "openai.com", "google.com", "bing.com", "wikipedia.org",
    "en.wikipedia.org", "youtube.com", "youtu.be", "mapbox.com", "twitter.com",
    "x.com", "facebook.com", "linkedin.com", "instagram.com", "reddit.com",
    "medium.com", "github.com", "utm.io",
}


def clean_domain(raw: str | None) -> str | None:
    """'https://www.Foo.com/bar?x=1' -> 'foo.com'; returns None if not domain-shaped."""
    if not raw or not isinstance(raw, str):
        return None
    d = raw.strip().lower()
    d = re.sub(r"^[a-z]+://", "", d)
    d = d.split("/")[0].split("?")[0].split("#")[0]
    d = d.split("@")[-1].split(":")[0]
    if d.startswith("www."):
        d = d[4:]
    d = d.strip(".")
    return d if _DOMAIN_RE.match(d) else None


def extract_text_domains(text: str) -> list[str]:
    """All distinct, cleaned domains linked inside the source text (order kept)."""
    seen, out = set(), []
    for m in _URL_RE.finditer(text):
        d = clean_domain(m.group(1))
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _registrable_label(domain: str) -> str:
    """Best-effort 'brand label' of a domain: foo in foo.com / foo.co.uk / sub.foo.com."""
    parts = domain.split(".")
    two_level_tlds = {"co", "com", "org", "net", "ac", "gov", "edu"}
    if len(parts) >= 3 and parts[-2] in two_level_tlds and len(parts[-1]) == 2:
        return parts[-3]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0]


def match_domain_from_text(entity_name: str, text_domains: list[str]) -> str | None:
    """Heuristic: if a URL in the document plausibly IS this entity's site, use it.
    e.g. 'Nationwide Crane Training' vs nationwidecranetraining.com."""
    slug = re.sub(r"[^a-z0-9]", "", normalize_name(entity_name))
    if not slug:
        return None
    for d in text_domains:
        if d in _IGNORED_URL_DOMAINS:
            continue
        label = _registrable_label(d).replace("-", "")
        if not label:
            continue
        if label == slug:
            return d
        if len(slug) >= 5 and slug in label:
            return d
        if len(label) >= 5 and label in slug:
            return d
        # acronym match: 'iti' vs Industrial Training International
        words = normalize_name(entity_name).split()
        if len(words) >= 2:
            acronym = "".join(w[0] for w in words if w and w[0].isalnum())
            if len(acronym) >= 3 and label == acronym:
                return d
    return None


def dns_exists(domain: str, timeout: float = DNS_TIMEOUT_SECONDS) -> bool | None:
    """True/False if resolvable/not; None if the check itself failed (no network)."""
    def _resolve():
        socket.getaddrinfo(domain, None)
        return True
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_resolve).result(timeout=timeout)
    except socket.gaierror:
        return False
    except Exception:
        return None  # timeout / sandboxed env: treat as "unknown", not "invalid"


def resolve_domain(
    entity_name: str,
    llm_domain: str | None,
    text_domains: list[str],
    verify_dns: bool = True,
) -> tuple[str | None, str, bool | None]:
    """Waterfall domain resolution.

    Returns (domain, source, dns_ok) where source is one of
    'curated' | 'text_url' | 'llm' | 'unresolved'.
    """
    hit = lookup(normalize_name(entity_name))
    if hit:
        return hit[1], "curated", None  # curated entries are trusted; skip DNS

    cleaned = clean_domain(llm_domain)

    # Strong signal: the document itself links to the entity's site.
    if cleaned and cleaned in text_domains:
        return cleaned, "text_url", None
    from_text = match_domain_from_text(entity_name, text_domains)
    if from_text:
        return from_text, "text_url", None

    if cleaned:
        dns_ok = dns_exists(cleaned) if verify_dns else None
        if dns_ok is False:
            # LLM invented a domain that does not exist -> refuse to report it.
            return None, "unresolved", False
        return cleaned, "llm", dns_ok

    return None, "unresolved", None


# --------------------------------------------------------------------------
# LLM call + JSON parsing
# --------------------------------------------------------------------------

def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # tolerate leading commentary before the first '{'
    i = raw.find("{")
    return raw[i:] if i > 0 else raw


def parse_llm_json(raw: str) -> list[dict]:
    data = json.loads(_strip_fences(raw))
    if isinstance(data, list):           # tolerate bare-array outputs
        entities = data
    else:
        entities = data.get("entities", [])
    if not isinstance(entities, list):
        raise ValueError("`entities` is not a list")
    return [e for e in entities if isinstance(e, dict)]


def llm_extract(text: str, client) -> list[dict]:
    user_msg = f"TEXT:\n\"\"\"\n{text}\n\"\"\""
    last_err: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            raw = client.complete(SYSTEM_PROMPT, user_msg)
            return parse_llm_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            user_msg += "\n\nREMINDER: respond with ONLY the JSON object described, nothing else."
        except Exception as e:  # network / rate limit
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise LLMError(f"LLM extraction failed after retries: {last_err}")


# --------------------------------------------------------------------------
# Core pipeline
# --------------------------------------------------------------------------

@dataclass
class _Record:
    key: str
    entity: str
    domain: str | None
    domain_source: str
    dns_ok: bool | None
    sentiments: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    first_pos: int = 1 << 30
    warnings: list[str] = field(default_factory=list)


def _coerce_sentiment(value, warnings: list[str]) -> str:
    s = str(value or "").strip().lower()
    if s in VALID_SENTIMENTS:
        return s
    warnings.append(f"invalid sentiment {value!r} coerced to neutral")
    return "neutral"


def _first_position(norm_text: str, candidate: dict, canonical: str) -> int:
    probes = ([canonical] + candidate.get("surface_forms", []) +
              candidate.get("evidence", []) + [candidate.get("name", "")])
    best = 1 << 30
    for p in probes:
        if not isinstance(p, str) or not p.strip():
            continue
        i = norm_text.find(_WS_RE.sub(" ", p.replace("*", "")).strip().lower())
        if i >= 0:
            best = min(best, i)
    return best


def extract_entities(
    text: str,
    *,
    client=None,
    verify_dns: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """Extract commercial entities from `text`.

    Returns a list of dicts ordered by first appearance:
        {"entity": str, "domain": str|None, "sentiment": "positive|negative|neutral"}
    With verbose=True each dict additionally carries evidence, confidence,
    domain_source, domain_dns_ok and any validation warnings.
    """
    if not text or not text.strip():
        return []
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    if client is None:
        client = default_client()

    candidates = llm_extract(text, client)
    norm_text = _normalize_for_matching(text)
    text_domains = extract_text_domains(text)

    records: dict[str, _Record] = {}
    for cand in candidates:
        warnings: list[str] = []
        name = cand.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        if cand.get("is_commercial") is False:
            continue
        # Layer 2: grounding - drop hallucinated entities.
        if not is_grounded(cand, norm_text):
            continue

        # Layer 3: canonicalization.
        norm = normalize_name(name)
        hit = lookup(norm)
        canonical = hit[0] if hit else name.strip()
        key = normalize_name(canonical)

        # Layer 4: domain waterfall.
        domain, source, dns_ok = resolve_domain(
            canonical, cand.get("domain"), text_domains, verify_dns=verify_dns
        )
        if source == "unresolved" and dns_ok is False:
            warnings.append("LLM-proposed domain failed DNS lookup and was discarded")

        sentiment = _coerce_sentiment(cand.get("sentiment"), warnings)
        try:
            confidence = float(cand.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        rec = records.get(key)
        if rec is None:
            rec = _Record(key=key, entity=canonical, domain=domain,
                          domain_source=source, dns_ok=dns_ok)
            records[key] = rec
        else:
            # keep the better-sourced domain on merge
            rank = {"curated": 3, "text_url": 2, "llm": 1, "unresolved": 0}
            if rank.get(source, 0) > rank.get(rec.domain_source, 0):
                rec.domain, rec.domain_source, rec.dns_ok = domain, source, dns_ok
        rec.sentiments.append(sentiment)
        rec.confidences.append(confidence)
        for e in cand.get("evidence", []):
            if isinstance(e, str) and e not in rec.evidence:
                rec.evidence.append(e)
        rec.warnings.extend(warnings)
        rec.first_pos = min(rec.first_pos, _first_position(norm_text, cand, canonical))

    # Layer 5: merge -> dominant sentiment (majority, tie-break by confidence).
    out = []
    for rec in sorted(records.values(), key=lambda r: r.first_pos):
        counts = {s: rec.sentiments.count(s) for s in set(rec.sentiments)}
        top = max(counts.values())
        leaders = [s for s, c in counts.items() if c == top]
        if len(leaders) == 1:
            sentiment = leaders[0]
        else:
            best_i = max(range(len(rec.sentiments)), key=lambda i: rec.confidences[i])
            sentiment = rec.sentiments[best_i] if rec.sentiments[best_i] in leaders else "neutral"

        item = {"entity": rec.entity, "domain": rec.domain, "sentiment": sentiment}
        if verbose:
            item.update({
                "confidence": round(sum(rec.confidences) / len(rec.confidences), 2),
                "evidence": rec.evidence[:3],
                "domain_source": rec.domain_source,
                "domain_dns_ok": rec.dns_ok,
                "warnings": rec.warnings,
            })
        out.append(item)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Extract commercial entities from text.")
    p.add_argument("text", nargs="?", help="input text (or use --file/--stdin)")
    p.add_argument("--file", help="read input text from a file")
    p.add_argument("--stdin", action="store_true", help="read input text from stdin")
    p.add_argument("--verbose", action="store_true",
                   help="include evidence, confidence and domain provenance")
    p.add_argument("--no-dns", action="store_true", help="skip DNS verification of domains")
    args = p.parse_args(argv)

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.stdin:
        text = sys.stdin.read()
    elif args.text:
        text = args.text
    else:
        p.error("provide input text, --file, or --stdin")

    result = extract_entities(text, verify_dns=not args.no_dns, verbose=args.verbose)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
