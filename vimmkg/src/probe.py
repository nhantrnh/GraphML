import json, os, re, collections, unicodedata, numpy as np, torch, random
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

random.seed(0); np.random.seed(0); torch.manual_seed(0)
D = os.path.expanduser("~/.cache/hanviet")
hv = json.load(open(f"{D}/hv_rad.json")); hv2ch, rad = hv["hv2ch"], hv["rad"]
ents = json.load(open(f"{D}/wikidata_vi_entities.json"))

def sylls(l):
    l = unicodedata.normalize("NFC", l.lower())
    return [s for s in re.split(r"[^a-zàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]+", l) if s]

# chỉ giữ âm KHÔNG mơ hồ về bộ thủ -> nhãn vàng chắc chắn
freq = collections.Counter(s for e in ents for s in sylls(e["label"]) if s in hv2ch)
gold = {}
for s in freq:
    rs = {rad[c] for c in hv2ch[s] if c in rad}
    if len(rs) == 1: gold[s] = rs.pop()
print(f"âm HV không mơ hồ bộ thủ: {len(gold)}/{len(freq)} type")

# giữ bộ thủ có >=20 mẫu để phân loại có nghĩa
cnt = collections.Counter(gold.values())
keep = {r for r,n in cnt.items() if n >= 20}
data = [(s,r) for s,r in gold.items() if r in keep]
print(f"sau lọc bộ thủ >=20 mẫu: {len(data)} âm, {len(keep)} lớp bộ thủ")
if len(data) < 100:
    print("QUÁ ÍT DỮ LIỆU — chuyển sang ngưỡng >=10")
    keep = {r for r,n in cnt.items() if n >= 10}
    data = [(s,r) for s,r in gold.items() if r in keep]
    print(f"  -> {len(data)} âm, {len(keep)} lớp")

tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
mdl = AutoModel.from_pretrained("vinai/phobert-base-v2").eval()

def emb(words, bs=64):
    out=[]
    for i in range(0,len(words),bs):
        b=words[i:i+bs]
        x=tok(b, return_tensors="pt", padding=True, truncation=True, max_length=8)
        with torch.no_grad(): h=mdl(**x).last_hidden_state
        m=x["attention_mask"].unsqueeze(-1).float()
        out.append(((h*m).sum(1)/m.sum(1)).numpy())
    return np.vstack(out)

words = [w for w,_ in data]; y = np.array([r for _,r in data])
X = emb(words)
print(f"embedding: {X.shape}")

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=0,stratify=y)
res={}
res["majority"] = DummyClassifier(strategy="most_frequent").fit(Xtr,ytr).score(Xte,yte)
clf = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1).fit(Xtr,ytr)
res["phobert_probe"] = clf.score(Xte,yte)
# đối chứng: embedding ngẫu nhiên -> đo mức "rò rỉ" do lớp phân loại
rng=np.random.RandomState(0); Xr=rng.randn(*X.shape).astype(np.float32)
Xrtr,Xrte,_,_ = train_test_split(Xr,y,test_size=0.25,random_state=0,stratify=y)
res["random_emb"] = LogisticRegression(max_iter=3000,n_jobs=-1).fit(Xrtr,ytr).score(Xrte,yte)
# đối chứng: chỉ ký tự bề mặt (char n-gram) -> ACL2018 cảnh báo baseline này
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
wtr=[words[i] for i in range(len(words))]
itr,ite = train_test_split(np.arange(len(words)),test_size=0.25,random_state=0,stratify=y)
pipe = make_pipeline(TfidfVectorizer(analyzer="char_wb",ngram_range=(1,3)), LogisticRegression(max_iter=3000))
pipe.fit([words[i] for i in itr], y[itr])
res["char_ngram"] = pipe.score([words[i] for i in ite], y[ite])

print("\n=== KẾT QUẢ PROBE: dự đoán BỘ THỦ từ biểu diễn âm Hán-Việt ===")
for k,v in res.items(): print(f"  {k:16} acc = {v*100:5.2f}%")
json.dump({"res":res,"n":len(data),"n_class":len(keep)}, open(f"{D}/probe_result.json","w"))
