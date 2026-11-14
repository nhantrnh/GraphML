#!/usr/bin/env python3
"""Cập nhật CUDA extension của fasterRCNN cho torch hiện đại.

Vì sao cần: `THC/THC.h` là header CUDA cũ của PyTorch, **bị gỡ hẳn từ torch
1.11**. Mã fasterRCNN trong ImparTail viết thời torch 1.x nên còn include nó, và
biên dịch trên bất kỳ torch nào từ 1.11 trở lên đều dừng ở:

    ROIAlign_cuda.cu:5:10: fatal error: THC/THC.h: No such file or directory

Đây không phải lỗi khó — chỉ có năm ký hiệu THC được dùng, tất cả đều có thay
thế thẳng trong ATen:

    THC/THC.h, THCDeviceUtils.cuh  -> không cần, ATen đã có
    THC/THCAtomics.cuh            -> ATen/cuda/Atomic.cuh
    THCCeilDiv(a, b)              -> impartail_ceil_div (tự khai báo, xem HEADERS)
    THCudaCheck(x)                -> AT_CUDA_CHECK(x)
    THCudaMalloc(state, n)        -> c10::cuda::CUDACachingAllocator::raw_alloc(n)
    THCudaFree(state, p)          -> c10::cuda::CUDACachingAllocator::raw_delete(p)
    THCState *state = ...lazyInitCUDA()  -> bỏ, không còn dùng

    python fix_thc.py <đường dẫn ImparTail>
"""
import os, re, sys

#: Tự định nghĩa ceil_div thay vì dùng at::ceil_div. Lý do: ATen/ceil_div.h chỉ
#: có từ torch 2.5; trên 2.4.1 (bản duy nhất còn hỗ trợ sm_60 của Tesla P100)
#: thì at::ceil_div chưa tồn tại, và biên dịch dừng ở
#:   error: namespace "at" has no member "ceil_div"
#: Hàm chỉ có một dòng nên tự khai báo là chắc chắn nhất, không phụ thuộc bản torch.
#: Header xác minh trực tiếp trên nguồn tag v2.4.1 của pytorch, không đoán:
#:   AT_CUDA_CHECK        -> aten/src/ATen/cuda/Exceptions.h   (KHÔNG phải CUDAException.h)
#:   getCurrentCUDAStream -> c10/cuda/CUDAStream.h             (KHÔNG phải CUDAContext.h)
#:   raw_alloc/raw_delete -> c10/cuda/CUDACachingAllocator.h
#:   gpuAtomicAdd         -> aten/src/ATen/cuda/Atomic.cuh
HEADERS = """#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace {
template <typename T>
__host__ __device__ __forceinline__ T impartail_ceil_div(T a, T b) {
  return (a + b - 1) / b;
}
}  // namespace"""

SUBS = [
    # include cũ -> bộ header hiện đại (chỉ chèn một lần cho tệp đầu tiên khớp)
    (re.compile(r"#include <THC/THC\.h>\n(#include <THC/THCAtomics\.cuh>\n)?"
                r"(#include <THC/THCDeviceUtils\.cuh>\n)?"), HEADERS + "\n"),
    (re.compile(r"#include <THC/[A-Za-z]+\.cuh>\n"), ""),
    (re.compile(r"\bTHCCeilDiv\b"), "impartail_ceil_div"),
    (re.compile(r"\bTHCudaCheck\b"), "AT_CUDA_CHECK"),
    (re.compile(r"THCudaMalloc\(state,\s*"),
     "c10::cuda::CUDACachingAllocator::raw_alloc("),
    (re.compile(r"THCudaFree\(state,\s*"),
     "c10::cuda::CUDACachingAllocator::raw_delete("),
    # dòng khởi tạo state không còn ý nghĩa
    (re.compile(r"^\s*THCState\s*\*\s*state\s*=.*$", re.M), ""),

    # ── API Tensor cũ, bỏ trong torch 1.x → 2.x ──────────────────────
    # Kiểm trực tiếp trên nguồn tag v2.4.1: TensorBase.h không còn khai báo
    # `.type()`, và `.data<T>()` đã thay bằng `data_ptr<T>()`.
    #   x.type().is_cuda()  -> x.is_cuda()          (25 + 18 chỗ)
    #   x.data<T>()         -> x.data_ptr<T>()      (26 chỗ)
    #   a.type() == b.type()-> a.scalar_type() == b.scalar_type()
    #   AT_CHECK            -> TORCH_CHECK
    (re.compile(r"\.type\(\)\.is_cuda\(\)"), ".is_cuda()"),
    (re.compile(r"(\w+)\.type\(\)\s*==\s*(\w+)\.type\(\)"),
     r"\1.scalar_type() == \2.scalar_type()"),
    (re.compile(r"AT_DISPATCH_FLOATING_TYPES\(\s*(\w+)\.type\(\)"),
     r"AT_DISPATCH_FLOATING_TYPES(\1.scalar_type()"),
    (re.compile(r"\.data<([A-Za-z_][A-Za-z0-9_]*)>\(\)"), r".data_ptr<\1>()"),
    (re.compile(r"\bAT_CHECK\b"), "TORCH_CHECK"),
]


def patch(path):
    src = open(path, encoding="utf-8").read()
    out = src
    for pat, rep in SUBS:
        out = pat.sub(rep, out)
    if out == src:
        return 0
    open(path, "w", encoding="utf-8").write(out)
    left = len(re.findall(r"\bTHC[A-Za-z_]*\b", out))
    print(f"  ✓ {os.path.basename(path):22} còn {left} ký hiệu THC")
    return 1


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    csrc = os.path.join(root, "fasterRCNN/lib/model/csrc")
    if not os.path.isdir(csrc):
        sys.exit(f"không thấy {csrc}")
    n = 0
    for dp, _, fn in os.walk(csrc):
        for f in fn:
            if f.endswith((".cu", ".cuh", ".h", ".cpp")):
                n += patch(os.path.join(dp, f))
    print(f"  đã vá {n} tệp")
    rest = []
    for dp, _, fn in os.walk(csrc):
        for f in fn:
            p = os.path.join(dp, f)
            if f.endswith((".cu", ".cuh", ".h", ".cpp")):
                for i, l in enumerate(open(p, encoding="utf-8"), 1):
                    if re.search(r"\bTHC[A-Za-z_]*\b", l):
                        rest.append(f"{f}:{i}: {l.strip()[:70]}")
    if rest:
        print("  CÒN SÓT:")
        for r in rest[:10]:
            print("    " + r)
    else:
        print("  không còn tham chiếu THC nào")
