"""Phép thử: ViMMKG có học được không? Dùng TransE thuần cấu trúc.

Nếu MRR ~ 0 thì bộ dữ liệu hỏng. Nếu MRR hợp lý (>0.15) thì đồ thị có tín hiệu
học được, và mọi kết quả đa mô thái sau này mới có nghĩa.
"""
import sys, os, json, time, argparse
APKGC = "/Users/nhantt/Downloads/MS_HCMUS/Study/HP3/GraphML/upstream/APKGC"
sys.path.insert(0, APKGC)
os.chdir(APKGC)   # đường dẫn benchmarks/ và embeddings/ đều tương đối
import torch, logging
from mmkgc.config import Trainer, Tester
from mmkgc.module.model import TransE
from mmkgc.module.loss import SigmoidLoss
from mmkgc.module.strategy import NegativeSampling
from mmkgc.data import TrainDataLoader, TestDataLoader

ap = argparse.ArgumentParser()
ap.add_argument("-dataset", default="ViMMKG")
ap.add_argument("-epoch", type=int, default=400)
ap.add_argument("-dim", type=int, default=128)
ap.add_argument("-batch", type=int, default=1024)
ap.add_argument("-neg", type=int, default=32)
ap.add_argument("-lr", type=float, default=1e-4)
ap.add_argument("-margin", type=float, default=12.0)
ap.add_argument("-seed", type=int, default=42)
ap.add_argument("-out", default="")
ap.add_argument("-early_stop", type=int, default=10000)   # tắt dừng sớm
# ba cờ Trainer bắt buộc (Trainer gắn với APKGC). add_noise=0 => bỏ nhánh gọi
# update_noise(), hàm chỉ APKGC mới có, TransE không có.
ap.add_argument("-add_noise", type=int, default=0)
ap.add_argument("-noise_update", default="none")
ap.add_argument("-save", default="/tmp/sanity_transe_ckpt")
a = ap.parse_args()
torch.manual_seed(a.seed)

t0 = time.time()
train = TrainDataLoader(
    in_path=f"./benchmarks/{a.dataset}/", batch_size=a.batch, threads=8,
    sampling_mode="cross", bern_flag=0, filter_flag=1, neg_ent=a.neg, neg_rel=0)
test = TestDataLoader(f"./benchmarks/{a.dataset}/", "link")
print(f"nạp dữ liệu {time.time()-t0:.0f}s: ent={train.get_ent_tot()} rel={train.get_rel_tot()}", flush=True)

model = TransE(ent_tot=train.get_ent_tot(), rel_tot=train.get_rel_tot(),
               dim=a.dim, p_norm=1, norm_flag=True, margin=a.margin)
ns = NegativeSampling(model=model, loss=SigmoidLoss(adv_temperature=2),
                      batch_size=train.get_batch_size())
# Trainer bắt buộc args (dùng args.dataset, args.early_stop) và tự tạo
# test_dataloader. save_steps=None => KHÔNG đánh giá giữa chừng => không dùng
# tập test để chọn mô hình. Đây chính là lỗi rò rỉ đã ghi ở C3, nên phép thử này
# cố ý tránh.
logging.basicConfig(level=logging.INFO, format="%(message)s")
lg = logging.getLogger("sanity")
tr = Trainer(args=a, logger=lg, model=ns, data_loader=train, train_times=a.epoch, alpha=a.lr,
             use_gpu=torch.cuda.is_available(), opt_method="adam",
             save_steps=None, checkpoint_dir=None)
t0 = time.time(); tr.run(); el = time.time()-t0
print(f"\nhuấn luyện {a.epoch} epoch trong {el/60:.1f} phút", flush=True)

te = Tester(model=model, data_loader=test, use_gpu=torch.cuda.is_available())
t0 = time.time()
mrr, mr, h10, h3, h1 = te.run_link_prediction(type_constrain=False)
print(f"đánh giá {(time.time()-t0)/60:.1f} phút")
res = dict(dataset=a.dataset, model="TransE", epoch=a.epoch, seed=a.seed,
           mrr=float(mrr), mr=float(mr), hit1=float(h1), hit3=float(h3), hit10=float(h10),
           train_min=el/60)
print(json.dumps(res, indent=2))
if a.out: json.dump(res, open(a.out, "w"), indent=2)
