"""Offline unit tests: the LLM layer is faked so the deterministic layers
(grounding, canonicalization, domain waterfall, merging) are verified without
an API key. Run:  pytest -q
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractor import (  # noqa: E402
    extract_entities, clean_domain, extract_text_domains, match_domain_from_text,
    normalize_name, resolve_domain,
)


class FakeLLM:
    """Returns a canned JSON payload regardless of input."""
    def __init__(self, entities):
        self.payload = json.dumps({"entities": entities})

    def complete(self, system, user):
        return self.payload


def E(name, sentiment, domain=None, evidence=None, **kw):
    d = {
        "name": name,
        "surface_forms": [name],
        "evidence": evidence or [name],
        "sentiment": sentiment,
        "sentiment_reason": "test",
        "domain": domain,
        "is_commercial": True,
        "confidence": kw.pop("confidence", 0.9),
    }
    d.update(kw)
    return d


# ---------------------------------------------------------------------------
# The four official assignment examples
# ---------------------------------------------------------------------------

def test_example_1_semrush_brightedge():
    text = ("We switched from Semrush to BrightEdge last year. Semrush was easier to get "
            "started with, but BrightEdge's data quality is far better for enterprise accounts.")
    fake = FakeLLM([
        E("Semrush", "negative", "semrush.com", ["switched from Semrush"]),
        E("BrightEdge", "positive", "brightedge.com", ["BrightEdge's data quality is far better"]),
    ])
    assert extract_entities(text, client=fake, verify_dns=False) == [
        {"entity": "Semrush", "domain": "semrush.com", "sentiment": "negative"},
        {"entity": "BrightEdge", "domain": "brightedge.com", "sentiment": "positive"},
    ]


def test_example_2_chatgpt():
    text = ("ChatGPT has completely changed how our team does research. We use it every day "
            "and it saves hours of work.")
    fake = FakeLLM([E("ChatGPT", "positive", "chatgpt.com", ["ChatGPT has completely changed"])])
    # curated map must override the LLM's chatgpt.com with the spec's openai.com
    assert extract_entities(text, client=fake, verify_dns=False) == [
        {"entity": "ChatGPT", "domain": "openai.com", "sentiment": "positive"},
    ]


def test_example_3_salesforce_hubspot():
    text = ("We evaluated Salesforce and HubSpot for our CRM rollout. Salesforce had more "
            "features but was too expensive and the implementation took months. HubSpot was "
            "easier to adopt and the support was great, though it lacked some advanced reporting.")
    fake = FakeLLM([
        E("Salesforce", "negative", "salesforce.com", ["Salesforce had more"]),
        E("HubSpot", "positive", "hubspot.com", ["HubSpot was"]),
    ])
    assert extract_entities(text, client=fake, verify_dns=False) == [
        {"entity": "Salesforce", "domain": "salesforce.com", "sentiment": "negative"},
        {"entity": "HubSpot", "domain": "hubspot.com", "sentiment": "positive"},
    ]


def test_example_4_aws_subdomain():
    text = ("AWS is the backbone of our infrastructure. It is reliable and the ecosystem is "
            "unmatched, but the billing is a nightmare to understand.")
    fake = FakeLLM([E("AWS", "positive", "amazon.com", ["AWS is the backbone"])])
    # curated map pins aws.amazon.com even if the LLM says amazon.com
    assert extract_entities(text, client=fake, verify_dns=False) == [
        {"entity": "AWS", "domain": "aws.amazon.com", "sentiment": "positive"},
    ]


# ---------------------------------------------------------------------------
# Anti-hallucination / validation layers
# ---------------------------------------------------------------------------

def test_hallucinated_entity_is_dropped():
    text = "We compared Semrush and Ahrefs for backlink analysis."
    fake = FakeLLM([
        E("Semrush", "neutral", "semrush.com", ["Semrush"]),
        E("Moz", "neutral", "moz.com", ["Moz offers keyword tools"]),  # not in text
    ])
    out = extract_entities(text, client=fake, verify_dns=False)
    assert [o["entity"] for o in out] == ["Semrush"]


def test_fabricated_domain_fails_dns_and_is_discarded():
    text = "Craneify totally changed our rigging workflow, highly recommend."
    fake = FakeLLM([
        E("Craneify", "positive",
          "craneify-this-domain-does-not-exist-zz9x.com", ["Craneify totally changed"]),
    ])
    out = extract_entities(text, client=fake, verify_dns=True, verbose=True)
    assert out[0]["entity"] == "Craneify"
    assert out[0]["domain"] is None          # refused to report a nonexistent domain
    assert out[0]["domain_source"] == "unresolved"


def test_non_commercial_and_invalid_sentiment():
    text = "OSHA requires certification. Slack is amazing for team chat."
    fake = FakeLLM([
        dict(E("OSHA", "neutral", "osha.gov", ["OSHA requires"]), is_commercial=False),
        E("Slack", "AMAZING!!", "slack.com", ["Slack is amazing"]),  # bad enum
    ])
    out = extract_entities(text, client=fake, verify_dns=False)
    assert out == [{"entity": "Slack", "domain": "slack.com", "sentiment": "neutral"}]


# ---------------------------------------------------------------------------
# Canonicalization & merging
# ---------------------------------------------------------------------------

def test_alias_merge_and_dominant_sentiment():
    text = ("Amazon Web Services powers everything we do; AWS support has been fantastic, "
            "although AWS billing confuses me.")
    fake = FakeLLM([
        E("Amazon Web Services", "positive", None, ["Amazon Web Services powers"], confidence=0.9),
        E("AWS", "positive", "aws.amazon.com", ["AWS support has been fantastic"], confidence=0.9),
        E("AWS", "negative", "aws.amazon.com", ["AWS billing confuses me"], confidence=0.6),
    ])
    out = extract_entities(text, client=fake, verify_dns=False)
    assert out == [{"entity": "AWS", "domain": "aws.amazon.com", "sentiment": "positive"}]


def test_rebrand_alias_twitter_to_x():
    text = "Twitter lost most of the features I loved."
    fake = FakeLLM([E("Twitter", "negative", "twitter.com", ["Twitter lost"])])
    out = extract_entities(text, client=fake, verify_dns=False)
    assert out == [{"entity": "X", "domain": "x.com", "sentiment": "negative"}]


def test_possessive_normalization():
    assert normalize_name("BrightEdge's") == "brightedge"
    assert normalize_name("McDonald's") == "mcdonald"  # both alias keys exist in the map
    assert normalize_name("Coca-Cola") == "coca cola"


# ---------------------------------------------------------------------------
# Domain resolution from in-text URLs
# ---------------------------------------------------------------------------

def test_domain_from_text_url_slug():
    text = ("* [Nationwide Crane Training](https://www.nationwidecranetraining.com/product/x/) "
            "provides live online rigger webinars.")
    fake = FakeLLM([E("Nationwide Crane Training", "neutral", None,
                      ["Nationwide Crane Training"])])
    out = extract_entities(text, client=fake, verify_dns=False, verbose=True)
    assert out[0]["domain"] == "nationwidecranetraining.com"
    assert out[0]["domain_source"] == "text_url"


def test_domain_from_text_url_acronym():
    text = ("[Industrial Training International (ITI)](https://www.iti.com/courses/rigging) "
            "is one of the most recognized training organizations.")
    fake = FakeLLM([E("Industrial Training International", "positive", None,
                      ["Industrial Training International"])])
    out = extract_entities(text, client=fake, verify_dns=False, verbose=True)
    assert out[0]["domain"] == "iti.com"


def test_llm_domain_confirmed_by_text_url():
    text = "I book everything through [this site](https://www.booking.com) these days."
    fake = FakeLLM([E("Booking.com", "positive", "booking.com", ["book everything through"])])
    out = extract_entities(text, client=fake, verify_dns=False, verbose=True)
    assert out[0]["domain"] == "booking.com"


# ---------------------------------------------------------------------------
# Helpers & edge cases
# ---------------------------------------------------------------------------

def test_clean_domain():
    assert clean_domain("https://www.Foo-Bar.com/x?y=1") == "foo-bar.com"
    assert clean_domain("aws.amazon.com") == "aws.amazon.com"
    assert clean_domain("not a domain") is None
    assert clean_domain(None) is None


def test_extract_text_domains_dedup_and_order():
    text = ("See [a](https://iti.com/a) and [b](https://www.iti.com/b) and "
            "[c](https://amcands.com/?utm_source=chatgpt.com)")
    assert extract_text_domains(text) == ["iti.com", "amcands.com"]


def test_empty_and_entityless_text():
    assert extract_entities("", client=FakeLLM([])) == []
    assert extract_entities("The weather was great and we hiked all afternoon.",
                            client=FakeLLM([])) == []


def test_output_order_follows_first_appearance():
    text = "HubSpot beat Salesforce for us: HubSpot was simpler."
    fake = FakeLLM([  # LLM returns them in the "wrong" order on purpose
        E("Salesforce", "negative", "salesforce.com", ["Salesforce"]),
        E("HubSpot", "positive", "hubspot.com", ["HubSpot beat"]),
    ])
    out = extract_entities(text, client=fake, verify_dns=False)
    assert [o["entity"] for o in out] == ["HubSpot", "Salesforce"]
