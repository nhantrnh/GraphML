#!/usr/bin/env python3
"""Dựng dan_y_apkgc.tex từ dan_y_apkgc.md (không cần pandoc).

Nguồn sự thật là tệp .md; tệp .tex sinh ra từ đó, nên sửa nội dung thì sửa
.md rồi chạy lại script này. Preamble lấy từ tệp .tex hiện có (phần trước
\\begin{document}), nên giữ nguyên font, màu và header.

Hỗ trợ đúng những gì dàn ý dùng: # / ## tiêu đề, > trích lời, **đậm**, *nghiêng*,
`mã`, $toán$, "ngoặc kép", và --- làm đường kẻ. Mỗi đoạn được gộp thành một
dòng trước khi chuyển, nên định dạng vắt qua nhiều dòng vẫn đúng.

Dùng: python md2tex_dany.py
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "dan_y_apkgc.md")
TEX = os.path.join(HERE, "dan_y_apkgc.tex")

TITLE = r"""\begin{document}

\begin{center}
  {\sffamily\LARGE\bfseries\color{hcmusblue} DÀN Ý TRÌNH BÀY --- 15 PHÚT}\\[8pt]
  {\sffamily\large\bfseries APKGC: Noise-enhanced Multi-Modal Knowledge Graph\\
   Completion with Attention Penalty}\\[6pt]
  {\small Y. Jian, X. Luo, Z. Li, M. Zhang, Y. Zhang, K. Xiao, X. Hou ---
   Hubei University --- AAAI 2025, tr. 15005--15013}\\[4pt]
  {\small Seminar GraphML --- HP3 --- ĐHQG-HCM, Trường ĐH Khoa học Tự nhiên}
\end{center}
\medskip
"""


def esc(t):
    # Giữ chỗ cho toán và mã, để định dạng **/*/"" vắt qua chúng vẫn khớp.
    raws = []

    def keep(m):
        v = m.group(0)
        if v.startswith("`"):
            c = v[1:-1]
            for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                         ("&", r"\&"), ("#", r"\#")):
                c = c.replace(a, b)
            v = r"\code{" + c + "}"
        raws.append(v)
        return f"\x00{len(raws) - 1}\x00"

    v = re.sub(r"(\$[^$]+\$|`[^`]+`)", keep, t)
    v = v.replace("\\", r"\textbackslash{}")
    for a, b in (("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("_", r"\_")):
        v = v.replace(a, b)
    v = re.sub(r'"(.+?)"', r"``\1''", v)
    v = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", v)
    v = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", v)
    for a, b in (("—", "---"), ("–", "--"), ("→", r"$\to$"), ("×", r"$\times$")):
        v = v.replace(a, b)
    return re.sub(r"\x00(\d+)\x00", lambda m: raws[int(m.group(1))], v)


def main():
    md = io.open(MD, encoding="utf-8").read()
    old = io.open(TEX, encoding="utf-8").read()
    pre = old[:old.index(r"\begin{document}")]
    pre = pre.replace("APKGC --- Báo cáo chuyên đề", "APKGC --- Dàn ý trình bày")

    # gộp thành khối: (loại, văn bản)
    blocks, cur, kind = [], [], None

    def flush():
        nonlocal cur, kind
        if cur:
            sep = "\n" if kind == "table" else " "
            blocks.append((kind, sep.join(x.strip() for x in cur)))
        cur, kind = [], None

    for L in md.split("\n"):
        if not L.strip():
            flush()
            continue
        if L.startswith("#"):
            flush()
            lvl = len(L) - len(L.lstrip("#"))
            blocks.append((f"h{lvl}", L[lvl:].strip()))
            continue
        if L.strip() == "---":
            flush()
            blocks.append(("hr", ""))
            continue
        if L.startswith("|"):
            if kind != "table":
                flush()
                kind = "table"
            cur.append(L)
            continue
        k = "quote" if L.startswith(">") else "para"
        if k != kind:
            flush()
            kind = k
        cur.append(L[1:] if k == "quote" else L)
    flush()

    body, seen_h1 = [], False
    for k, t in blocks:
        if k == "h1":
            if not seen_h1:
                seen_h1 = True
                continue
            body.append(r"\newpage" + "\n" + r"\section*{" + esc(t) + "}")
        elif k == "h2":
            body.append(r"\section*{" + esc(t) + "}")
        elif k == "hr":
            body.append(r"\medskip\noindent\rule{\linewidth}{0.3pt}\medskip")
        elif k == "table":
            rows = [r for r in t.split("\n") if not set(r.replace("|", "").strip()) <= set("-: ")]
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            n = len(cells[0])
            if n == 2:
                spec = "@{}lp{0.72\\linewidth}@{}"
            elif n == 3:  # bảng dài: cột cuối tự xuống dòng
                spec = "@{}lp{0.16\\linewidth}p{0.6\\linewidth}@{}"
            else:
                spec = "@{}" + "l" * n + "@{}"
            out = [r"\begin{center}\small", r"\begin{tabular}{" + spec + "}", r"\toprule"]
            out.append(" & ".join(r"\textbf{" + esc(c) + "}" for c in cells[0]) + r" \\ \midrule")
            for row in cells[1:]:
                out.append(" & ".join(esc(c) for c in row) + r" \\")
            out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
            body.append("\n".join(out))
        elif k == "quote":
            body.append(r"\begin{quote}\itshape " + esc(t) + r"\end{quote}")
        else:
            body.append(esc(t) + "\n")
    doc = pre + TITLE + "\n\n".join(body) + "\n\\end{document}\n"
    # Times New Roman thiếu các glyph này; thay bằng lệnh LaTeX tương đương.
    for ch, rep in (("⇒", r"\ensuremath{\Rightarrow}"), ("①", "(1)"), ("②", "(2)")):
        doc = doc.replace(ch, rep)
    io.open(TEX, "w", encoding="utf-8").write(doc)
    print(f"đã ghi {os.path.basename(TEX)}: {sum(k.startswith('h') for k, _ in blocks)} tiêu đề, "
          f"{sum(k == 'quote' for k, _ in blocks)} trích lời")


if __name__ == "__main__":
    main()
