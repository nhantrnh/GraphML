"""Bản sửa cho `noise_ratio` của APKGC: từ tham số chết thành tham số học được.

Vấn đề gốc (mmkgc/module/model/APKGC.py, L131):

    noise_mask = torch.rand(n) < self.noise_ratio

Phép so sánh trả về bool tensor nên không có đường gradient — `noise_ratio`
đứng yên ở giá trị khởi tạo 0.2 suốt quá trình huấn luyện, dù bài báo mô tả
nó là "adaptively tuned during the training process" (Eq. 5).

Cách sửa dùng ở đây là concrete relaxation (Maddison et al., ICLR 2017;
Jang et al., ICLR 2017), theo đúng khuôn mẫu Concrete Dropout (Gal, Hron,
Kendall, NeurIPS 2017) — vốn được đề xuất chính xác để học xác suất Bernoulli
thay vì grid search.

Module này cài đặt hai biến thể và một baseline để so sánh sòng phẳng:

    FixedBernoulliMask   — tái hiện hành vi thật của APKGC (đối chứng)
    ConcreteMask         — relaxation liên tục, mask mềm
    StraightThroughMask  — mask cứng khi forward, gradient chảy khi backward
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_sigmoid(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(torch.log(torch.tensor(p / (1 - p))))


class FixedBernoulliMask(nn.Module):
    """Đối chứng: tái hiện đúng hành vi APKGC hiện tại.

    `p` khai báo là nn.Parameter nhưng chỉ dùng trong phép so sánh, nên không
    bao giờ nhận gradient. Giữ nguyên ở đây để phần thực nghiệm có baseline
    trung thực với bản gốc.
    """

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))

    def forward(self, n: int, device=None) -> torch.Tensor:
        return (torch.rand(n, device=device) < self.p).float()

    @property
    def rate(self) -> float:
        return float(self.p)


class ConcreteMask(nn.Module):
    """Concrete / Gumbel-Softmax relaxation cho mask Bernoulli.

    Thay mask rời rạc bằng sigmoid của một logit có nhiễu Gumbel, chia cho
    nhiệt độ. Khi nhiệt độ giảm, mask mềm tiến dần về mask cứng, nhưng vẫn
    khả vi theo `logit` ở mọi thời điểm.

    Tuỳ chọn `regularize` thêm một số hạng entropy nhẹ, giữ tỉ lệ nhiễu không
    trôi về 0 (mô hình sẽ luôn muốn tắt nhiễu vì nhiễu làm loss huấn luyện
    tăng — đây là điểm cần lưu ý khi diễn giải kết quả).
    """

    def __init__(self, p_init: float = 0.2, temperature: float = 0.5,
                 eps: float = 1e-7):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(_inv_sigmoid(p_init)))
        self.temperature = temperature
        self.eps = eps

    def forward(self, n: int, device=None) -> torch.Tensor:
        if not self.training:
            return (torch.rand(n, device=device) < torch.sigmoid(self.logit)).float()

        u = torch.rand(n, device=device).clamp(self.eps, 1 - self.eps)
        noise = torch.log(u) - torch.log1p(-u)          # logistic noise
        return torch.sigmoid((self.logit + noise) / self.temperature)

    @property
    def rate(self) -> float:
        return float(torch.sigmoid(self.logit))

    def entropy_penalty(self) -> torch.Tensor:
        """Entropy của Bernoulli(p). Cộng vào loss với hệ số âm nhỏ để giữ p
        không sụp về 0."""
        p = torch.sigmoid(self.logit).clamp(self.eps, 1 - self.eps)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


class StraightThroughMask(ConcreteMask):
    """Biến thể straight-through (Bengio et al., 2013).

    Forward dùng mask nhị phân đúng như bản gốc, nên hành vi huấn luyện không
    đổi về mặt ngữ nghĩa; backward dùng gradient của mask mềm. Đây là lựa chọn
    ít xâm lấn nhất khi muốn giữ nguyên động lực học của mô hình.
    """

    def forward(self, n: int, device=None) -> torch.Tensor:
        soft = super().forward(n, device)
        if not self.training:
            return soft
        hard = (soft > 0.5).float()
        return hard + soft - soft.detach()


def apply_noise(embeddings: torch.Tensor,
                mean: torch.Tensor,
                std: torch.Tensor,
                mask: torch.Tensor,
                mask_ratio: torch.Tensor) -> torch.Tensor:
    """Tiêm nhiễu theo đúng công thức APKGC nhưng dùng mask liên tục.

    Bản gốc dùng boolean indexing (`emb[noise_mask] = ...`), vốn không truyền
    gradient về mask. Ở đây thay bằng phép nội suy có trọng số, giữ nguyên ngữ
    nghĩa khi mask nhận giá trị 0/1 nhưng cho phép gradient chảy ngược.
    """
    noise = mean + std * torch.randn_like(embeddings)
    perturbed = (1.0 - mask_ratio) * embeddings + mask_ratio * noise
    m = mask.unsqueeze(-1)
    return m * perturbed + (1.0 - m) * embeddings


# ----------------------------------------------------------------------------
# Kiểm chứng: tham số có thực sự nhận gradient không
# ----------------------------------------------------------------------------

def _check(name: str, module: nn.Module) -> Tuple[bool, float]:
    torch.manual_seed(0)
    emb = torch.randn(256, 32)
    mean, std = emb.mean(0), emb.std(0)
    mask_ratio = nn.Parameter(torch.tensor(0.7))
    proj = nn.Linear(32, 32)

    module.train()
    params = list(module.parameters())
    for p in params:
        p.grad = None

    mask = module(emb.shape[0])
    out = apply_noise(emb, mean, std, mask, mask_ratio)
    loss = proj(out).pow(2).mean()
    loss.backward()

    grads = [p.grad for p in params]
    got = any(g is not None for g in grads)
    norm = max((float(g.norm()) for g in grads if g is not None), default=0.0)

    status = "CÓ gradient" if got else "KHÔNG có gradient"
    print(f"  {name:<24} p = {module.rate:.4f}   {status:<18} "
          f"‖grad‖ = {norm:.3e}" if got else
          f"  {name:<24} p = {module.rate:.4f}   {status}")
    return got, norm


def _train_toy(module: nn.Module, steps: int = 300, lr: float = 0.05) -> float:
    """Huấn luyện đồ chơi: mục tiêu ưu tiên giữ tín hiệu gốc, nên tỉ lệ nhiễu
    tối ưu nên giảm. Dùng để xác nhận tham số thực sự di chuyển."""
    torch.manual_seed(0)
    emb = torch.randn(256, 32)
    mean, std = emb.mean(0), emb.std(0)
    target = emb.clone()

    mask_ratio = nn.Parameter(torch.tensor(0.7))
    opt = torch.optim.Adam(list(module.parameters()) + [mask_ratio], lr=lr)

    module.train()
    for _ in range(steps):
        opt.zero_grad()
        mask = module(emb.shape[0])
        out = apply_noise(emb, mean, std, mask, mask_ratio)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()

    return module.rate


if __name__ == "__main__":
    print("=" * 74)
    print("KIỂM CHỨNG ĐƯỜNG GRADIENT")
    print("=" * 74)
    _check("FixedBernoulli (gốc)", FixedBernoulliMask(0.2))
    _check("Concrete", ConcreteMask(0.2))
    _check("StraightThrough", StraightThroughMask(0.2))

    print()
    print("=" * 74)
    print("THAM SỐ CÓ DI CHUYỂN KHÔNG (300 bước, mục tiêu phạt nhiễu)")
    print("=" * 74)
    for name, mod in [("FixedBernoulli (gốc)", FixedBernoulliMask(0.2)),
                      ("Concrete", ConcreteMask(0.2)),
                      ("StraightThrough", StraightThroughMask(0.2))]:
        before = mod.rate
        after = _train_toy(mod)
        delta = after - before
        moved = "đứng yên" if abs(delta) < 1e-6 else f"dịch {delta:+.4f}"
        print(f"  {name:<24} {before:.4f} -> {after:.4f}   ({moved})")
    print()
    print("  Bản gốc không thay đổi vì không có gradient. Hai biến thể còn lại")
    print("  giảm tỉ lệ nhiễu, đúng như mục tiêu đồ chơi mong đợi.")
