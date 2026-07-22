"""Curated dictionary of well-known brands: alias -> (canonical display name, primary domain).

Why this exists
---------------
LLMs resolve domains from parametric knowledge, which is usually right for famous
brands but not guaranteed, and it can drift between runs. For high-frequency
brands we want *deterministic* answers, and we want to pin policy decisions
(e.g. the assignment expects ChatGPT -> openai.com even though chatgpt.com
exists, and AWS -> aws.amazon.com rather than amazon.com). A small curated map
is the cheapest, most reliable way to lock those in. Everything not in the map
falls through to URL-evidence and LLM resolution (see extractor.resolve_domain).

Keys are matched after normalization: lowercase, punctuation stripped,
whitespace collapsed (see extractor.normalize_name).
"""

# alias -> (canonical_name, domain)
KNOWN_ENTITIES: dict[str, tuple[str, str]] = {
    # --- Assignment examples (must be deterministic) ---
    "semrush": ("Semrush", "semrush.com"),
    "brightedge": ("BrightEdge", "brightedge.com"),
    "chatgpt": ("ChatGPT", "openai.com"),          # per assignment spec
    "chat gpt": ("ChatGPT", "openai.com"),
    "salesforce": ("Salesforce", "salesforce.com"),
    "hubspot": ("HubSpot", "hubspot.com"),
    "aws": ("AWS", "aws.amazon.com"),
    "amazon web services": ("AWS", "aws.amazon.com"),
    "amazons cloud": ("AWS", "aws.amazon.com"),
    "amazon cloud": ("AWS", "aws.amazon.com"),

    # --- SEO / marketing tools (BrightEdge's space) ---
    "ahrefs": ("Ahrefs", "ahrefs.com"),
    "moz": ("Moz", "moz.com"),
    "conductor": ("Conductor", "conductor.com"),
    "screaming frog": ("Screaming Frog", "screamingfrog.co.uk"),
    "clearscope": ("Clearscope", "clearscope.io"),
    "surfer seo": ("Surfer", "surferseo.com"),
    "surfer": ("Surfer", "surferseo.com"),
    "mailchimp": ("Mailchimp", "mailchimp.com"),
    "brevo": ("Brevo", "brevo.com"),
    "google analytics": ("Google Analytics", "analytics.google.com"),
    "google search console": ("Google Search Console", "search.google.com"),

    # --- Big tech companies ---
    "google": ("Google", "google.com"),
    "alphabet": ("Alphabet", "abc.xyz"),
    "microsoft": ("Microsoft", "microsoft.com"),
    "apple": ("Apple", "apple.com"),
    "amazon": ("Amazon", "amazon.com"),
    "meta": ("Meta", "meta.com"),
    "facebook": ("Facebook", "facebook.com"),
    "instagram": ("Instagram", "instagram.com"),
    "whatsapp": ("WhatsApp", "whatsapp.com"),
    "netflix": ("Netflix", "netflix.com"),
    "nvidia": ("NVIDIA", "nvidia.com"),
    "intel": ("Intel", "intel.com"),
    "amd": ("AMD", "amd.com"),
    "ibm": ("IBM", "ibm.com"),
    "oracle": ("Oracle", "oracle.com"),
    "sap": ("SAP", "sap.com"),
    "samsung": ("Samsung", "samsung.com"),
    "sony": ("Sony", "sony.com"),
    "tesla": ("Tesla", "tesla.com"),

    # --- Cloud / dev infra ---
    "azure": ("Azure", "azure.microsoft.com"),
    "microsoft azure": ("Azure", "azure.microsoft.com"),
    "gcp": ("Google Cloud", "cloud.google.com"),
    "google cloud": ("Google Cloud", "cloud.google.com"),
    "google cloud platform": ("Google Cloud", "cloud.google.com"),
    "digitalocean": ("DigitalOcean", "digitalocean.com"),
    "digital ocean": ("DigitalOcean", "digitalocean.com"),
    "hetzner": ("Hetzner", "hetzner.com"),
    "cloudflare": ("Cloudflare", "cloudflare.com"),
    "vercel": ("Vercel", "vercel.com"),
    "github": ("GitHub", "github.com"),
    "gitlab": ("GitLab", "gitlab.com"),
    "docker": ("Docker", "docker.com"),
    "databricks": ("Databricks", "databricks.com"),
    "snowflake": ("Snowflake", "snowflake.com"),
    "stripe": ("Stripe", "stripe.com"),
    "paypal": ("PayPal", "paypal.com"),
    "shopify": ("Shopify", "shopify.com"),
    "twilio": ("Twilio", "twilio.com"),

    # --- AI ---
    "openai": ("OpenAI", "openai.com"),
    "gpt-4": ("ChatGPT", "openai.com"),
    "gpt4": ("ChatGPT", "openai.com"),
    "anthropic": ("Anthropic", "anthropic.com"),
    "claude": ("Claude", "claude.ai"),
    "gemini": ("Gemini", "gemini.google.com"),
    "google gemini": ("Gemini", "gemini.google.com"),
    "copilot": ("GitHub Copilot", "github.com"),
    "github copilot": ("GitHub Copilot", "github.com"),
    "perplexity": ("Perplexity", "perplexity.ai"),
    "midjourney": ("Midjourney", "midjourney.com"),
    "hugging face": ("Hugging Face", "huggingface.co"),
    "huggingface": ("Hugging Face", "huggingface.co"),

    # --- SaaS / productivity ---
    "slack": ("Slack", "slack.com"),
    "zoom": ("Zoom", "zoom.us"),
    "notion": ("Notion", "notion.so"),
    "figma": ("Figma", "figma.com"),
    "figjam": ("FigJam", "figma.com"),
    "miro": ("Miro", "miro.com"),
    "airtable": ("Airtable", "airtable.com"),
    "asana": ("Asana", "asana.com"),
    "trello": ("Trello", "trello.com"),
    "jira": ("Jira", "atlassian.com"),
    "confluence": ("Confluence", "atlassian.com"),
    "atlassian": ("Atlassian", "atlassian.com"),
    "linkedin": ("LinkedIn", "linkedin.com"),
    "youtube": ("YouTube", "youtube.com"),
    "spotify": ("Spotify", "spotify.com"),
    "dropbox": ("Dropbox", "dropbox.com"),
    "adobe": ("Adobe", "adobe.com"),
    "photoshop": ("Photoshop", "adobe.com"),
    "canva": ("Canva", "canva.com"),
    "zendesk": ("Zendesk", "zendesk.com"),
    "intercom": ("Intercom", "intercom.com"),
    "monday": ("monday.com", "monday.com"),
    "monday com": ("monday.com", "monday.com"),

    # --- Products mapped to parent domain (policy: product without its own
    #     standalone primary domain resolves to the operating company) ---
    "iphone": ("iPhone", "apple.com"),
    "macbook": ("MacBook", "apple.com"),
    "ipad": ("iPad", "apple.com"),
    "windows": ("Windows", "microsoft.com"),
    "office 365": ("Microsoft 365", "microsoft.com"),
    "microsoft 365": ("Microsoft 365", "microsoft.com"),
    "excel": ("Microsoft Excel", "microsoft.com"),
    "gmail": ("Gmail", "google.com"),
    "android": ("Android", "android.com"),

    # --- Rebrands / renames (alias both old and new to the current identity) ---
    "twitter": ("X", "x.com"),
    "x": ("X", "x.com"),
    "convertkit": ("Kit", "kit.com"),

    # --- Consumer / retail / travel ---
    "walmart": ("Walmart", "walmart.com"),
    "target": ("Target", "target.com"),
    "costco": ("Costco", "costco.com"),
    "nike": ("Nike", "nike.com"),
    "adidas": ("Adidas", "adidas.com"),
    "coca cola": ("Coca-Cola", "coca-cola.com"),
    "cocacola": ("Coca-Cola", "coca-cola.com"),
    "pepsi": ("Pepsi", "pepsi.com"),
    "mcdonalds": ("McDonald's", "mcdonalds.com"),
    "mcdonald": ("McDonald's", "mcdonalds.com"),
    "starbucks": ("Starbucks", "starbucks.com"),
    "uber": ("Uber", "uber.com"),
    "lyft": ("Lyft", "lyft.com"),
    "airbnb": ("Airbnb", "airbnb.com"),
    "booking": ("Booking.com", "booking.com"),
    "booking com": ("Booking.com", "booking.com"),
    "expedia": ("Expedia", "expedia.com"),
    "delta": ("Delta Air Lines", "delta.com"),
    "delta air lines": ("Delta Air Lines", "delta.com"),
    "united airlines": ("United Airlines", "united.com"),
    "marriott": ("Marriott", "marriott.com"),
}


def lookup(normalized_name: str) -> tuple[str, str] | None:
    """Return (canonical_name, domain) for a normalized alias, else None."""
    return KNOWN_ENTITIES.get(normalized_name)
