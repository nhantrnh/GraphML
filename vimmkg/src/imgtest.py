import json, os, time, urllib.request, urllib.parse, random
UA="ViMMKG-research/0.1 (academic; your-email@example.com)"
D=os.path.expanduser("~/.cache/hanviet")
ents=json.load(open(f"{D}/wikidata_vi_entities.json"))
random.seed(0); samp=random.sample(ents,60)

# lấy URL ảnh + giấy phép qua Wikidata API
def wbget(qids):
    u="https://www.wikidata.org/w/api.php?"+urllib.parse.urlencode(
        {"action":"wbgetentities","ids":"|".join(qids),"props":"claims","format":"json"})
    req=urllib.request.Request(u,headers={"User-Agent":UA})
    with urllib.request.urlopen(req,timeout=60) as r: return json.load(r)

fnames={}
for i in range(0,len(samp),25):
    b=[e["qid"] for e in samp[i:i+25]]
    d=wbget(b)
    for q,v in d.get("entities",{}).items():
        c=v.get("claims",{}).get("P18",[])
        if c: fnames[q]=c[0]["mainsnak"]["datavalue"]["value"]
    time.sleep(1)
print(f"có tên file ảnh: {len(fnames)}/{len(samp)}")

# tải thử qua Special:FilePath (bản thu nhỏ 224px)
ok=fail=0; tot=0; t0=time.time(); lic=[]
os.makedirs(f"{D}/img",exist_ok=True)
for q,fn in list(fnames.items())[:30]:
    url="https://commons.wikimedia.org/wiki/Special:FilePath/"+urllib.parse.quote(fn)+"?width=224"
    try:
        req=urllib.request.Request(url,headers={"User-Agent":UA})
        with urllib.request.urlopen(req,timeout=45) as r: data=r.read()
        open(f"{D}/img/{q}.jpg","wb").write(data); ok+=1; tot+=len(data)
    except Exception as e:
        fail+=1; print(f"  lỗi {q}: {type(e).__name__} {str(e)[:50]}")
el=time.time()-t0
print(f"\ntải: {ok} ok / {fail} lỗi trong {el:.0f}s = {el/max(ok,1):.2f}s/ảnh")
print(f"kích thước TB: {tot/max(ok,1)/1024:.0f} KB @224px")
print(f"=> ước tính 15.000 ảnh: {15000*el/max(ok,1)/3600:.1f} giờ tuần tự, {15000*tot/max(ok,1)/1e9:.2f} GB")
