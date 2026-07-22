"""Minimal demo service for the extractor.

    export ANTHROPIC_API_KEY=...
    uvicorn app:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000 for an interactive page,
or call the API directly:

    curl -s localhost:8000/extract -H 'content-type: application/json' \
         -d '{"text": "We switched from Semrush to BrightEdge last year."}'
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from extractor import extract_entities

app = FastAPI(title="Entity Extraction Demo", version="1.0")


class ExtractRequest(BaseModel):
    text: str
    verbose: bool = False
    verify_dns: bool = True


@app.post("/extract")
def extract(req: ExtractRequest):
    try:
        entities = extract_entities(req.text, verify_dns=req.verify_dns, verbose=req.verbose)
        return {"ok": True, "entities": entities}
    except Exception as e:  # surface a readable error to the demo page
        return {"ok": False, "error": str(e), "entities": []}


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Entity Extraction Demo</title>
<style>
  :root { --ink:#1b1f24; --sub:#5a6470; --line:#d9dee5; --accent:#0b5fff;
          --pos:#0a7a3d; --neg:#b3261e; --neu:#5a6470; --chip:#f2f5f9; }
  * { box-sizing:border-box; }
  body { margin:0; font:16px/1.55 "Iowan Old Style", Georgia, serif; color:var(--ink);
         background:#fbfcfd; }
  main { max-width:880px; margin:0 auto; padding:48px 24px 96px; }
  h1 { font-size:30px; margin:0 0 4px; letter-spacing:-.01em; }
  .sub { color:var(--sub); font-family:ui-monospace,Menlo,Consolas,monospace; font-size:13px;
         margin-bottom:28px; }
  textarea { width:100%; min-height:170px; padding:14px 16px; font:15px/1.5 ui-monospace,
             Menlo,Consolas,monospace; border:1px solid var(--line); border-radius:8px;
             background:#fff; resize:vertical; }
  textarea:focus { outline:2px solid var(--accent); outline-offset:1px; border-color:transparent; }
  .row { display:flex; gap:12px; align-items:center; margin:14px 0 30px; flex-wrap:wrap; }
  button { font:15px/1 inherit; padding:11px 22px; border-radius:8px; border:1px solid var(--ink);
           background:var(--ink); color:#fff; cursor:pointer; }
  button:disabled { opacity:.5; cursor:wait; }
  .ghost { background:transparent; color:var(--ink); }
  label.opt { font-size:14px; color:var(--sub); display:flex; gap:6px; align-items:center; }
  table { width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line);
          border-radius:8px; overflow:hidden; }
  th, td { text-align:left; padding:12px 16px; border-top:1px solid var(--line);
           font-size:15px; vertical-align:top; }
  thead th { border-top:0; background:var(--chip); font-family:ui-monospace,Menlo,monospace;
             font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--sub); }
  .s { font-family:ui-monospace,Menlo,monospace; font-size:13px; padding:3px 10px;
       border-radius:99px; border:1px solid currentColor; }
  .positive { color:var(--pos); } .negative { color:var(--neg); } .neutral { color:var(--neu); }
  .domain { font-family:ui-monospace,Menlo,monospace; font-size:14px; }
  .empty, .err { color:var(--sub); padding:22px 4px; }
  .err { color:var(--neg); white-space:pre-wrap; }
  details { margin-top:22px; } summary { cursor:pointer; color:var(--sub); font-size:14px; }
  pre { background:#0f1720; color:#dce6f2; padding:16px; border-radius:8px; overflow:auto;
        font-size:13px; }
</style></head><body><main>
<h1>Entity Extraction</h1>
<div class="sub">extract_entities(text) &rarr; entity &middot; domain &middot; sentiment</div>
<textarea id="t">We switched from Semrush to BrightEdge last year. Semrush was easier to get started with, but BrightEdge's data quality is far better for enterprise accounts.</textarea>
<div class="row">
  <button id="go">Extract entities</button>
  <button class="ghost" id="sample">Load another sample</button>
  <label class="opt"><input type="checkbox" id="v"> show evidence</label>
</div>
<div id="out"></div>
<details><summary>Raw JSON response</summary><pre id="raw">-</pre></details>
<script>
const samples = [
 "ChatGPT has completely changed how our team does research. We use it every day and it saves hours of work.",
 "We evaluated Salesforce and HubSpot for our CRM rollout. Salesforce had more features but was too expensive and the implementation took months. HubSpot was easier to adopt and the support was great, though it lacked some advanced reporting.",
 "AWS is the backbone of our infrastructure. It is reliable and the ecosystem is unmatched, but the billing is a nightmare to understand.",
 "After two outages we migrated off DigitalOcean to Hetzner and have not looked back.",
 "* [Industrial Training International (ITI)](https://www.iti.com/courses/rigging) - one of the industry's most recognized rigging training organizations."
];
let si = 0;
document.getElementById('sample').onclick = () => {
  document.getElementById('t').value = samples[si++ % samples.length];
};
document.getElementById('go').onclick = async () => {
  const btn = document.getElementById('go'), out = document.getElementById('out');
  btn.disabled = true; out.innerHTML = '<div class="empty">Extracting…</div>';
  try {
    const r = await fetch('/extract', {method:'POST',
      headers:{'content-type':'application/json'},
      body: JSON.stringify({text: document.getElementById('t').value,
                            verbose: document.getElementById('v').checked})});
    const data = await r.json();
    document.getElementById('raw').textContent = JSON.stringify(data, null, 2);
    if (!data.ok) { out.innerHTML = '<div class="err">'+data.error+'</div>'; return; }
    if (!data.entities.length) { out.innerHTML = '<div class="empty">No commercial entities found.</div>'; return; }
    const showEv = document.getElementById('v').checked;
    let html = '<table><thead><tr><th>entity</th><th>domain</th><th>sentiment</th>'
             + (showEv ? '<th>evidence</th>' : '') + '</tr></thead><tbody>';
    for (const e of data.entities) {
      html += '<tr><td><strong>'+e.entity+'</strong></td>'
            + '<td class="domain">'+(e.domain ?? '<em>unresolved</em>')+'</td>'
            + '<td><span class="s '+e.sentiment+'">'+e.sentiment+'</span></td>'
            + (showEv ? '<td>'+ (e.evidence||[]).map(q=>'&ldquo;'+q+'&rdquo;').join('<br>') +'</td>' : '')
            + '</tr>';
    }
    out.innerHTML = html + '</tbody></table>';
  } catch (err) { out.innerHTML = '<div class="err">'+err+'</div>'; }
  finally { btn.disabled = false; }
};
</script></main></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
