"""
Phantom Parameter Audit
=======================

Phát hiện các tham số được khai báo `nn.Parameter(requires_grad=True)` nhưng
KHÔNG BAO GIỜ nhận gradient trong thực tế — thường vì chúng chỉ tham gia vào
các phép toán không khả vi (so sánh, indexing, argmax, ...).

Đây là một dạng lỗi âm thầm: mô hình *tuyên bố* học được một tham số, nhưng
tham số đó đứng yên ở giá trị khởi tạo suốt quá trình huấn luyện.

Cách dùng:
    from audit_gradient import audit
    report = audit(model, forward_fn)
    report.summary()

Trong đó `forward_fn()` là một hàm không tham số, thực hiện một lượt
forward + tính loss và trả về scalar loss.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import torch
import torch.nn as nn


@dataclass
class ParamVerdict:
    name: str
    shape: tuple
    numel: int
    requires_grad: bool
    got_grad: bool
    grad_norm: Optional[float]
    verdict: str          # "learned" | "phantom" | "frozen" | "zero-grad"
    note: str = ""


@dataclass
class AuditReport:
    params: List[ParamVerdict] = field(default_factory=list)
    n_steps: int = 0

    @property
    def phantoms(self) -> List[ParamVerdict]:
        return [p for p in self.params if p.verdict == "phantom"]

    @property
    def zero_grads(self) -> List[ParamVerdict]:
        return [p for p in self.params if p.verdict == "zero-grad"]

    def summary(self) -> None:
        learned = [p for p in self.params if p.verdict == "learned"]
        frozen = [p for p in self.params if p.verdict == "frozen"]

        print("=" * 74)
        print(f"PHANTOM PARAMETER AUDIT  ({self.n_steps} bước forward/backward)")
        print("=" * 74)
        print(f"  Tổng số tham số      : {len(self.params)}")
        print(f"  Học được bình thường : {len(learned)}")
        print(f"  Đóng băng có chủ ý   : {len(frozen)}")
        print(f"  Gradient bằng 0      : {len(self.zero_grads)}")
        print(f"  PHANTOM              : {len(self.phantoms)}")
        print()

        if self.phantoms:
            print("-" * 74)
            print("PHANTOM — khai báo learnable nhưng KHÔNG nhận gradient")
            print("-" * 74)
            for p in self.phantoms:
                print(f"  ✗ {p.name:<44} shape={str(p.shape):<14} grad=None")
            print()
            print("  Các tham số trên đứng yên ở giá trị khởi tạo suốt quá trình")
            print("  huấn luyện. Nếu bài báo mô tả chúng là 'adaptive', 'learnable'")
            print("  hay 'automatically tuned' thì mô tả đó không khớp cài đặt.")
            print()

        if self.zero_grads:
            print("-" * 74)
            print("GRADIENT BẰNG 0 — có đường gradient nhưng giá trị triệt tiêu")
            print("-" * 74)
            for p in self.zero_grads:
                print(f"  ~ {p.name:<44} ‖grad‖ = {p.grad_norm:.3e}")
            print()
            print("  Có thể do khởi tạo, do detach ở đâu đó, hoặc chỉ là ngẫu nhiên")
            print("  ở bước này. Nên chạy lại với nhiều bước để phân biệt.")
            print()

        if not self.phantoms and not self.zero_grads:
            print("  Không phát hiện tham số bất thường.")
            print()

    def as_markdown(self) -> str:
        rows = ["| Tham số | Shape | requires_grad | Nhận gradient | Kết luận |",
                "|---|---|---|---|---|"]
        for p in sorted(self.params, key=lambda x: (x.verdict != "phantom", x.name)):
            mark = {"phantom": "**PHANTOM**", "learned": "học được",
                    "frozen": "đóng băng", "zero-grad": "grad = 0"}[p.verdict]
            rows.append(
                f"| `{p.name}` | {tuple(p.shape)} | {p.requires_grad} | "
                f"{'có' if p.got_grad else 'KHÔNG'} | {mark} |")
        return "\n".join(rows)


def audit(model: nn.Module,
          forward_fn: Callable[[], torch.Tensor],
          n_steps: int = 3,
          zero_tol: float = 1e-12) -> AuditReport:
    """Chạy `n_steps` lượt forward/backward và ghi nhận tham số nào nhận gradient.

    Dùng nhiều bước vì một số tham số chỉ tham gia vào nhánh tính toán được
    kích hoạt ngẫu nhiên (dropout, sampling), nên một bước đơn lẻ có thể
    kết luận nhầm.
    """
    model.train()

    tracked: Dict[str, dict] = {
        name: {"param": p, "ever_grad": False, "max_norm": 0.0}
        for name, p in model.named_parameters()
    }

    for _ in range(n_steps):
        model.zero_grad(set_to_none=True)
        loss = forward_fn()
        if not torch.is_tensor(loss) or loss.dim() != 0:
            raise ValueError("forward_fn phải trả về một scalar tensor (loss).")
        loss.backward()

        for name, info in tracked.items():
            g = info["param"].grad
            if g is not None:
                info["ever_grad"] = True
                info["max_norm"] = max(info["max_norm"], float(g.norm()))

    report = AuditReport(n_steps=n_steps)
    for name, info in tracked.items():
        p = info["param"]
        got = info["ever_grad"]
        norm = info["max_norm"] if got else None

        if not p.requires_grad:
            verdict = "frozen"
        elif not got:
            verdict = "phantom"
        elif norm is not None and norm <= zero_tol:
            verdict = "zero-grad"
        else:
            verdict = "learned"

        report.params.append(ParamVerdict(
            name=name, shape=tuple(p.shape), numel=p.numel(),
            requires_grad=p.requires_grad, got_grad=got,
            grad_norm=norm, verdict=verdict))

    return report


# ----------------------------------------------------------------------------
# Minh hoạ: tái hiện đúng lỗi tìm thấy trong APKGC
# ----------------------------------------------------------------------------

class _APKGCNoisePattern(nn.Module):
    """Tái hiện cách APKGC tiêm nhiễu (mmkgc/module/model/APKGC.py, L42-135).

    Hai tham số khai báo hoàn toàn giống nhau, nhưng chỉ một cái học được:
      - mask_ratio  tham gia phép NHÂN      -> có gradient
      - noise_ratio tham gia phép SO SÁNH   -> không có gradient
    """

    def __init__(self, n_ent=64, dim=16):
        super().__init__()
        self.mask_ratio = nn.Parameter(torch.tensor(0.7))
        self.noise_ratio = nn.Parameter(torch.tensor(0.2))
        self.proj = nn.Linear(dim, dim)
        self.register_buffer("emb", torch.randn(n_ent, dim))

    def forward(self):
        emb = self.emb.clone()
        mean, std = emb.mean(0), emb.std(0)

        # Phép so sánh: trả về bool tensor, cắt đứt đường gradient tới noise_ratio
        noise_mask = torch.rand(emb.shape[0], device=emb.device) < self.noise_ratio

        selected = emb[noise_mask]
        noise = mean + std * torch.randn_like(selected)

        # Phép nhân: gradient chảy được tới mask_ratio
        emb[noise_mask] = (1.0 - self.mask_ratio) * selected + self.mask_ratio * noise

        return self.proj(emb).pow(2).mean()


def _demo() -> AuditReport:
    torch.manual_seed(0)
    model = _APKGCNoisePattern()
    report = audit(model, model.forward, n_steps=5)
    report.summary()
    return report


if __name__ == "__main__":
    _demo()
