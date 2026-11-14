import json,os,time,urllib.request,urllib.parse,random,statistics
UA="ViMMKG-research/0.1 (academic; your-email@example.com)"
D=os.path.expanduser("~/.cache/hanviet")
ents=json.load(open(f"{D}/wikidata_vi_entities.json"))
random.seed(1); samp=random.sample(ents,200)

# 1) description@vi có bao nhiêu
def wbget(qids):
    u="https://www.wikidata.org/w/api.php?"+urllib.parse.urlencode(
      {"action":"wbgetentities","ids":"|".join(qids),"props":"descriptions|sitelinks","languages":"vi","format":"json"})
    with urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":UA}),timeout=60) as r:
        return json.load(r)
desc=0; titles={}
for i in range(0,len(samp),50):
    d=wbget([e["qid"] for e in samp[i:i+50]])
    for q,v in d.get("entities",{}).items():
        if v.get("descriptions",{}).get("vi"): desc+=1
        sl=v.get("sitelinks",{}).get("viwiki")
        if sl: titles[q]=sl["title"]
    time.sleep(1)
print(f"có description@vi : {desc}/{len(samp)} = {desc/len(samp)*100:.1f}%")
print(f"có bài viwiki     : {len(titles)}/{len(samp)} = {len(titles)/len(samp)*100:.1f}%")

# 2) đoạn mở đầu viwiki
lens=[]; empty=0
tl=list(titles.items())
for i in range(0,len(tl),20):
    b=tl[i:i+20]
    u="https://vi.wikipedia.org/w/api.php?"+urllib.parse.urlencode(
      {"action":"query","prop":"extracts","exintro":"1","explaintext":"1",
       "titles":"|".join(t for _,t in b),"format":"json","redirects":"1"})
    try:
        with urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":UA}),timeout=90) as r:
            d=json.load(r)
        for _,pg in d.get("query",{}).get("pages",{}).items():
            ex=(pg.get("extract") or "").strip()
            if ex: lens.append(len(ex))
            else: empty+=1
    except Exception as e: print("  lỗi lô:",type(e).__name__)
    time.sleep(1)
print(f"\nđoạn mở đầu lấy được: {len(lens)}/{len(titles)} ({len(lens)/max(len(titles),1)*100:.1f}%), rỗng {empty}")
if lens:
    lens.sort()
    print(f"  độ dài ký tự: trung vị {statistics.median(lens):.0f}, TB {statistics.mean(lens):.0f}")
    print(f"  phân vị 10/90: {lens[len(lens)//10]} / {lens[len(lens)*9//10]}")
    print(f"  ngắn hơn 100 ký tự: {sum(1 for l in lens if l<100)} ({sum(1 for l in lens if l<100)/len(lens)*100:.0f}%)")
