#!/usr/bin/env bash
# Render one report figure to PNG with the report's fonts and packages (xelatex), without
# rebuilding the report. Adapted from the paper-figures skill's preview.sh.
# Usage: figures/preview.sh <figure.tex> <width-cm> [dpi] [outdir]
set -euo pipefail
fig=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
width=${2:?width in cm}; dpi=${3:-220}
out=${4:-${TMPDIR:-/tmp}/report-figures}
style=$(dirname "$fig")/style.tex
name=$(basename "$fig" .tex)
mkdir -p "$out"
cat > "$out/$name-preview.tex" <<EOF
\documentclass{article}
\usepackage[paperwidth=$(echo "$width + 0.6" | bc)cm,paperheight=20cm,margin=0.3cm]{geometry}
\usepackage{fontspec}\setmainfont{Times New Roman}
\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\usepackage{amsmath,amssymb,xcolor,tikz,pgfplots,fontawesome5}
\usetikzlibrary{positioning,arrows.meta,backgrounds,calc,shapes.geometric}
\input{$style}\pagestyle{empty}
\begin{document}\noindent\resizebox{${width}cm}{!}{\input{$fig}}\end{document}
EOF
(cd "$out" && xelatex -interaction=nonstopmode "$name-preview.tex" >/dev/null 2>&1) || true
if grep -q '^!' "$out/$name-preview.log"; then grep -A3 '^!' "$out/$name-preview.log" | head -20; exit 1; fi
pdftoppm -r "$dpi" -singlefile -png -f 1 -l 1 "$out/$name-preview.pdf" "$out/$name"
# trim the blank page around the figure
python3 -c "from PIL import Image, ImageChops as C; im=Image.open('$out/$name.png').convert('RGB'); b=C.difference(im, Image.new('RGB', im.size, (255,255,255))).getbbox(); im.crop((max(b[0]-8,0), max(b[1]-8,0), b[2]+8, b[3]+8)).save('$out/$name.png')"
echo "$out/$name.png"
