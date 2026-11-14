"""Chuyển báo cáo Markdown sang LaTeX.

Viết riêng cho cấu trúc của report.md. Đặt trong thư mục dự án để không mất
khi thư mục tạm bị dọn.

Dùng: python md2tex.py report.md report.tex
"""

import re
import sys

MATH = {
    "→": r"\rightarrow", "←": r"\leftarrow", "⇒": r"\Rightarrow",
    "⇔": r"\Leftrightarrow", "≈": r"\approx", "×": r"\times",
    "∼": r"\sim", "≥": r"\geq", "≤": r"\leq", "≠": r"\neq",
    "±": r"\pm", "‖": r"\Vert", "⊙": r"\odot", "∈": r"\in",
    "∘": r"\circ", "μ": r"\mu", "σ": r"\sigma", "θ": r"\theta",
    "φ": r"\phi", "ρ": r"\rho", "λ": r"\lambda", "ε": r"\varepsilon",
    "Σ": r"\Sigma", "γ": r"\gamma", "β": r"\beta", "τ": r"\tau",
    "δ": r"\delta", "α": r"\alpha", "Δ": r"\Delta",
}


def esc(t: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        t = t.replace(a, b)
    return t


def inline(t: str) -> str:
    """Định dạng nội tuyến. Đánh dấu vùng in đậm trước khi tách theo dấu nháy
    ngược, vì một cụm in đậm có thể chứa mã bên trong."""
    B_OPEN, B_CLOSE, I_OPEN, I_CLOSE = "\x01", "\x02", "\x03", "\x04"
    t = re.sub(r"\*\*(.+?)\*\*", B_OPEN + r"\1" + B_CLOSE, t)
    t = re.sub(r"(?<![\*\x01\x02])\*([^*\x01\x02]+)\*(?![\*\x01\x02])",
               I_OPEN + r"\1" + I_CLOSE, t)

    out = []
    for p in re.split(r"(`[^`]+`)", t):
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(r"\code{" + esc(p[1:-1]) + "}")
            continue
        p = esc(p)
        p = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", p)
        p = p.replace("—", "---").replace("–", "--")
        for ch, cmd in MATH.items():
            p = p.replace(ch, f"${cmd}$")
        out.append(p)

    res = "".join(out)
    res = res.replace(B_OPEN, r"\textbf{").replace(B_CLOSE, "}")
    res = res.replace(I_OPEN, r"\emph{").replace(I_CLOSE, "}")
    return res


def convert(md: str) -> str:
    lines = md.split("\n")
    out, i, n = [], 0, len(md.split("\n"))

    while i < n:
        ln = lines[i]

        if ln.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            out.append(r"\begin{lstlisting}")
            out.extend(buf)
            out.append(r"\end{lstlisting}")
            continue

        if ln.strip() == "---":
            i += 1; continue

        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            lvl = len(m.group(1))
            title = re.sub(r"^\d+(\.\d+)*\.?\s*", "", m.group(2).strip())
            cmd = {1: "section", 2: "section", 3: "subsection",
                   4: "subsubsection"}[lvl]
            out.append(f"\\{cmd}{{{inline(title)}}}")
            i += 1; continue

        if ln.strip().startswith("|") and i + 1 < n and \
           re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            hdr = [c.strip() for c in ln.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in
                             lines[i].strip().strip("|").split("|")])
                i += 1
            ncol = len(hdr)
            numeric = [all(re.match(r"^[\d.,\s±+\-−]*$", r[c]) and r[c]
                           for r in rows if c < len(r))
                       for c in range(ncol)]
            spec = "l" + "".join("r" if numeric[c] else "l"
                                 for c in range(1, ncol))
            out.append(r"\begin{center}\small")
            out.append(r"\begin{tabular}{@{}" + spec + r"@{}}")
            out.append(r"\toprule")
            out.append(" & ".join(r"\textbf{" + inline(h) + "}"
                                  for h in hdr) + r"\\")
            out.append(r"\midrule")
            for r in rows:
                r = (r + [""] * ncol)[:ncol]
                out.append(" & ".join(inline(c) for c in r) + r"\\")
            out.append(r"\bottomrule")
            out.append(r"\end{tabular}")
            out.append(r"\end{center}")
            continue

        if re.match(r"^\d+\.\s+", ln):
            items = []
            while i < n and re.match(r"^\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i])); i += 1
            out.append(r"\begin{enumerate}\itemsep2pt")
            out.extend(r"\item " + inline(it) for it in items)
            out.append(r"\end{enumerate}")
            continue

        if re.match(r"^[-*]\s+", ln):
            items = []
            while i < n and re.match(r"^[-*]\s+", lines[i]):
                items.append(re.sub(r"^[-*]\s+", "", lines[i])); i += 1
            out.append(r"\begin{itemize}\itemsep2pt")
            out.extend(r"\item " + inline(it) for it in items)
            out.append(r"\end{itemize}")
            continue

        if ln.strip() == "":
            i += 1; continue

        # Đoạn mở đầu bằng nhãn in đậm: tách riêng để cặp ** không bị cắt ngang
        if ln.lstrip().startswith("**"):
            out.append(inline(ln.strip()))
            out.append("")
            i += 1; continue

        buf = []
        while i < n and lines[i].strip() != "" and \
              not re.match(r"^(#{1,4}\s|```|\||[-*]\s|\d+\.\s)", lines[i]) and \
              lines[i].strip() != "---":
            buf.append(lines[i].strip()); i += 1
        if buf:
            out.append(inline(" ".join(buf)))
            out.append("")

    return "\n".join(out)


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

\usepackage{fontspec}
\usepackage{polyglossia}
\setmainlanguage{vietnamese}
\setotherlanguage{english}
\setmainfont{Times New Roman}
\setsansfont{Helvetica Neue}
\setmonofont{Menlo}[Scale=0.82]

\usepackage[margin=2.4cm,top=2.6cm,bottom=2.4cm]{geometry}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}

\definecolor{hcmusblue}{HTML}{1B3A5C}
\definecolor{contextblue}{HTML}{2E86C1}
\definecolor{codebg}{HTML}{F5F7FA}
\definecolor{coderule}{HTML}{2E5C8A}
\definecolor{codetext}{HTML}{A03030}

\titleformat{\section}{\sffamily\large\bfseries\color{hcmusblue}}
            {\thesection.}{0.6em}{}[\vspace{2pt}\hrule]
\titleformat{\subsection}{\sffamily\normalsize\bfseries\color{contextblue}}
            {\thesubsection}{0.6em}{}
\titleformat{\subsubsection}{\sffamily\small\bfseries}
            {\thesubsubsection}{0.6em}{}
\titlespacing*{\section}{0pt}{14pt}{6pt}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}

\setlist[itemize]{leftmargin=1.3em, topsep=3pt, itemsep=2pt}
\setlist[enumerate]{leftmargin=1.5em, topsep=3pt, itemsep=2pt}

\newcommand{\code}[1]{{\ttfamily\small\color{codetext}#1}}

\lstset{
  basicstyle=\ttfamily\scriptsize,
  backgroundcolor=\color{codebg},
  frame=leftline, framerule=1.5pt, rulecolor=\color{coderule},
  framesep=6pt, xleftmargin=8pt,
  breaklines=true, breakatwhitespace=false,
  showstringspaces=false, columns=fullflexible,
}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\color{gray}APKGC --- Báo cáo chuyên đề}
\fancyhead[R]{\footnotesize\color{gray}\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\begin{document}
"""

TITLE = r"""
\begin{center}
  {\sffamily\LARGE\bfseries\color{hcmusblue} BÁO CÁO CHUYÊN ĐỀ}\\[8pt]
  {\sffamily\large\bfseries APKGC: Noise-enhanced Multi-Modal Knowledge Graph\\
   Completion with Attention Penalty}\\[10pt]
  {\small Seminar GraphML --- Học phần HP3}\\[2pt]
  {\small ĐHQG-HCM --- Trường Đại học Khoa học Tự nhiên}
\end{center}

\vspace{4pt}
\begin{center}\small
\begin{tabular}{@{}ll@{}}
  \toprule
  \textbf{Hội nghị} & The 39th AAAI Conference on Artificial Intelligence
                      (AAAI-25), Oral, tr. 15005--15013\\
  \textbf{Tác giả}  & Yue Jian, Xiangyu Luo, Zhifei Li, Miao Zhang, Yan Zhang,
                      Kui Xiao, Xiaoju Hou\\
                    & Hubei University, Trung Quốc\\
  \textbf{Mã nguồn} & \texttt{github.com/HubuKG/APKGC}\\
  \bottomrule
\end{tabular}
\end{center}

\vspace{6pt}
"""


def main():
    src = open(sys.argv[1], encoding="utf-8").read()
    i = src.find("## 1. Lý do chọn bài báo")
    body = src[i:] if i > 0 else src
    tex = PREAMBLE + TITLE + convert(body) + "\n\\end{document}\n"
    open(sys.argv[2], "w", encoding="utf-8").write(tex)
    print(f"đã ghi {sys.argv[2]}")


if __name__ == "__main__":
    main()
