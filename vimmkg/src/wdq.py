import json, time, urllib.parse, urllib.request, sys, os
UA = "ViMMKG-research/0.1 (academic; your-email@example.com)"
EP = "https://query.wikidata.org/sparql"

CLASSES = [
    ("human",        "wd:Q5",       True),
    ("settlement",   "wd:Q486972",  True),
    ("taxon",        "wd:Q16521",   False),
    ("organization", "wd:Q43229",   True),
    ("building",     "wd:Q41176",   True),
    ("work",         "wd:Q386724",  True),
    ("geo",          "wd:Q618123",  True),
]

def q(cls, sub, limit):
    p = "wdt:P31/wdt:P279*" if sub else "wdt:P31"
    return f"""SELECT DISTINCT ?item ?label WHERE {{
  ?item {p} {cls} ; wdt:P18 ?img .
  ?sl schema:about ?item ; schema:isPartOf <https://vi.wikipedia.org/> .
  ?item rdfs:label ?label . FILTER(lang(?label)="vi")
}} LIMIT {limit}"""

def run(sparql, timeout=90):
    url = EP + "?" + urllib.parse.urlencode({"query": sparql, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

seen, out = set(), []
for name, cls, sub in CLASSES:
    for attempt, (s, lim) in enumerate([(sub, 3000), (False, 3000), (False, 1000)]):
        try:
            t0 = time.time()
            d = run(q(cls, s, lim))
            rows = d["results"]["bindings"]
            n = 0
            for b in rows:
                qid = b["item"]["value"].rsplit("/", 1)[-1]
                if qid in seen: continue
                seen.add(qid); out.append({"qid": qid, "label": b["label"]["value"], "class": name}); n += 1
            print(f"  {name:13} +{n:5} mới (trả {len(rows)}) trong {time.time()-t0:.0f}s", flush=True)
            break
        except Exception as e:
            print(f"  {name:13} thử {attempt+1} lỗi: {type(e).__name__} {str(e)[:70]}", flush=True)
            time.sleep(3)
    time.sleep(2)

P = os.path.expanduser("~/.cache/hanviet/wikidata_vi_entities.json")
json.dump(out, open(P, "w"), ensure_ascii=False)
print(f"\nTỔNG: {len(out):,} thực thể riêng biệt -> {P}")
