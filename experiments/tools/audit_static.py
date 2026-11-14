"""Phát hiện tĩnh các tham số có nguy cơ không nhận gradient.

Bản audit động (`audit_gradient.py`) cần dựng được mô hình và chạy một lượt
forward, nên phải chuẩn bị dữ liệu và phụ thuộc cho từng repo. Bản tĩnh này
chỉ phân tích cây cú pháp, nên chạy được trên bất kỳ mã nguồn nào mà không
cần cài đặt gì.

Nó tìm các tham số khai báo bằng `nn.Parameter` rồi kiểm tra xem chúng được
dùng ở đâu. Một tham số chỉ xuất hiện trong các phép toán không khả vi —
so sánh, indexing, ép kiểu, `.item()` — thì không thể nhận gradient.

Kết quả là *nghi vấn*, không phải kết luận. Mọi trường hợp phát hiện được nên
xác nhận lại bằng bản audit động trước khi đưa vào báo cáo.

Cách dùng:
    python audit_static.py <đường-dẫn-repo> [<đường-dẫn-repo> ...]
"""

import argparse
import ast
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set


# Các phép toán cắt đứt đường gradient khi tham số đi qua chúng.
COMPARISONS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
BLOCKING_CALLS = {"item", "int", "float", "bool", "round", "argmax", "argmin",
                  "nonzero", "detach", "numpy", "tolist"}


@dataclass
class ParamUse:
    line: int
    context: str          # "comparison" | "arithmetic" | "call:<tên>" | "other"
    snippet: str


@dataclass
class ParamInfo:
    name: str
    decl_line: int
    decl_snippet: str
    frozen: bool = False   # requires_grad=False ngay tại khai báo hoặc gán sau đó
    uses: List[ParamUse] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.frozen:
            return "dong-bang"
        if not self.uses:
            return "khong-dung"
        kinds = {u.context for u in self.uses}
        if "arithmetic" in kinds:
            return "co-gradient"
        if kinds <= {"comparison"} or all(
                k == "comparison" or k.startswith("call:") for k in kinds):
            return "NGHI-VAN"
        return "chua-ro"


class ParamVisitor(ast.NodeVisitor):
    """Thu thập khai báo `self.x = nn.Parameter(...)` và mọi nơi dùng `self.x`."""

    def __init__(self, src_lines: List[str]):
        self.lines = src_lines
        self.params: Dict[str, ParamInfo] = {}
        self._pending_uses: List[tuple] = []   # (tên, dòng, ngữ cảnh)
        self._frozen_later: Set[str] = set()

    def _line(self, n: int) -> str:
        return self.lines[n - 1].strip() if 0 < n <= len(self.lines) else ""

    # --- khai báo ---------------------------------------------------------
    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Call):
            fn = node.value.func
            fname = getattr(fn, "attr", getattr(fn, "id", ""))
            if fname == "Parameter":
                frozen = any(
                    kw.arg == "requires_grad"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is False
                    for kw in node.value.keywords)
                for t in node.targets:
                    if (isinstance(t, ast.Attribute)
                            and isinstance(t.value, ast.Name)
                            and t.value.id == "self"):
                        self.params[t.attr] = ParamInfo(
                            name=t.attr, decl_line=node.lineno,
                            decl_snippet=self._line(node.lineno),
                            frozen=frozen)

        # Bat `self.x.requires_grad = False` dat sau khai bao
        for t in node.targets:
            if (isinstance(t, ast.Attribute) and t.attr == "requires_grad"
                    and isinstance(t.value, ast.Attribute)
                    and isinstance(t.value.value, ast.Name)
                    and t.value.value.id == "self"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is False):
                self._frozen_later.add(t.value.attr)
        self.generic_visit(node)

    # --- nơi sử dụng ------------------------------------------------------
    def visit_Compare(self, node: ast.Compare) -> None:
        if any(isinstance(op, COMPARISONS) for op in node.ops):
            for name in self._attrs_in(node):
                self._pending_uses.append((name, node.lineno, "comparison"))
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        for name in self._attrs_in(node):
            self._pending_uses.append((name, node.lineno, "arithmetic"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fname = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if fname in BLOCKING_CALLS:
            for name in self._attrs_in(node):
                self._pending_uses.append((name, node.lineno, f"call:{fname}"))
        self.generic_visit(node)

    @staticmethod
    def _attrs_in(node: ast.AST) -> Set[str]:
        found = set()
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"):
                found.add(sub.attr)
        return found

    def finish(self) -> Dict[str, ParamInfo]:
        for name in self._frozen_later:
            if name in self.params:
                self.params[name].frozen = True
        for name, line, ctx in self._pending_uses:
            if name in self.params:
                self.params[name].uses.append(
                    ParamUse(line=line, context=ctx, snippet=self._line(line)))
        return self.params


def scan_file(path: str) -> Dict[str, ParamInfo]:
    try:
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
    except (SyntaxError, UnicodeDecodeError):
        return {}
    v = ParamVisitor(src.splitlines())
    v.visit(tree)
    return v.finish()


def scan_repo(repo: str) -> List[tuple]:
    """Trả về danh sách (đường-dẫn-tương-đối, ParamInfo) có nghi vấn."""
    findings = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs
                   if d not in {".git", "__pycache__", "build", "dist"}]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            for info in scan_file(p).values():
                findings.append((os.path.relpath(p, repo), info))
    return findings


def report(repo: str) -> Dict[str, int]:
    name = os.path.basename(repo.rstrip("/"))
    findings = scan_repo(repo)

    counts = {"co-gradient": 0, "NGHI-VAN": 0, "khong-dung": 0,
              "chua-ro": 0, "dong-bang": 0}
    for _, info in findings:
        counts[info.verdict] += 1

    print("=" * 74)
    print(f"{name}  —  {len(findings)} tham số khai báo bằng nn.Parameter")
    print("=" * 74)
    print(f"  có đường gradient rõ ràng : {counts['co-gradient']}")
    print(f"  nghi vấn                  : {counts['NGHI-VAN']}")
    print(f"  không thấy dùng ở đâu     : {counts['khong-dung']}")
    print(f"  chưa xác định             : {counts['chua-ro']}")
    print(f"  đóng băng có chủ ý        : {counts['dong-bang']}")

    flagged = [(f, i) for f, i in findings if i.verdict == "NGHI-VAN"]
    if flagged:
        print()
        print("  NGHI VẤN — chỉ xuất hiện trong phép toán không khả vi:")
        for f, info in flagged:
            print(f"    {f}:{info.decl_line}  self.{info.name}")
            print(f"      khai báo: {info.decl_snippet}")
            for u in info.uses[:3]:
                print(f"      dòng {u.line} ({u.context}): {u.snippet[:70]}")

    unused = [(f, i) for f, i in findings if i.verdict == "khong-dung"]
    if unused:
        print()
        print("  KHÔNG THẤY DÙNG trong cùng tệp (có thể dùng ở nơi khác):")
        for f, info in unused[:8]:
            print(f"    {f}:{info.decl_line}  self.{info.name}")

    print()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    args = ap.parse_args()

    total = {}
    for r in args.repos:
        if not os.path.isdir(r):
            print(f"Bỏ qua {r}: không phải thư mục", file=sys.stderr)
            continue
        total[os.path.basename(r.rstrip("/"))] = report(r)

    if len(total) > 1:
        print("=" * 74)
        print("TỔNG HỢP")
        print("=" * 74)
        print(f"  {'repo':<16}{'tổng':>7}{'gradient':>10}{'nghi vấn':>11}{'không dùng':>13}")
        for name, c in total.items():
            tot = sum(c.values())
            print(f"  {name:<16}{tot:>7}{c['co-gradient']:>10}"
                  f"{c['NGHI-VAN']:>11}{c['khong-dung']:>13}")
        print()
        print("  Kết quả cột 'nghi vấn' cần xác nhận bằng audit động trước khi")
        print("  đưa vào báo cáo. Phân tích tĩnh không theo được luồng gọi hàm")
        print("  giữa các tệp, nên có thể bỏ sót hoặc báo nhầm.")


if __name__ == "__main__":
    main()
