from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "paper_draft.md"
TEX_PATH = ROOT / "rag_shield_ieee_final_holdout.tex"


FIGURE_FILES = {
    "Fig. 1": "figures/pipeline.drawio.png",
    "Fig. 2": "figures/fig2_roc_pr.png",
    "Fig. 3": "figures/fig3_confusion_attribution.png",
    "Fig. 4": "figures/fig4_latency.png",
    "Fig. 5": "figures/fig5_method_comparison.png",
}


def normalize_unicode(s: str) -> str:
    replacements = {
        "—": "--",
        "–": "--",
        "×": r"$\times$",
        "→": r"$\to$",
        "←": r"$\leftarrow$",
        "≥": r"$\geq$",
        "≤": r"$\leq$",
        "≈": r"$\approx$",
        "τ": r"$\tau$",
        "ε": r"$\epsilon$",
        "Φ": r"$\Phi$",
        "ｉｇｎｏｒｅ": "ignore (full-width Unicode)",
        "Schölkopf": r"Scholkopf",
        "Guzmán": r"Guzman",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def latex_escape(s: str) -> str:
    s = normalize_unicode(s)
    chars = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(chars.get(c, c) for c in s)


def inline_latex(text: str) -> str:
    text = normalize_unicode(text.strip())
    pieces = re.split(r"(\$[^$]+\$)", text)
    out: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if piece.startswith("$") and piece.endswith("$"):
            out.append(normalize_unicode(piece))
            continue

        placeholders: dict[str, str] = {}

        def stash(value: str) -> str:
            key = f"@@P{len(placeholders)}@@"
            placeholders[key] = value
            return key

        piece = re.sub(
            r"`([^`]+)`",
            lambda m: stash(r"\texttt{" + latex_escape(m.group(1)) + "}"),
            piece,
        )
        piece = re.sub(
            r"\*\*([^*]+)\*\*",
            lambda m: stash(r"\textbf{" + inline_latex(m.group(1)) + "}"),
            piece,
        )
        piece = re.sub(
            r"\*([^*]+)\*",
            lambda m: stash(r"\emph{" + inline_latex(m.group(1)) + "}"),
            piece,
        )

        escaped = latex_escape(piece)
        for key, value in placeholders.items():
            escaped = escaped.replace(latex_escape(key), value)
        escaped = re.sub(r"\[(\d+)\]", r"\\cite{ref\1}", escaped)
        out.append(escaped)
    return "".join(out)


def strip_numbered_heading(title: str) -> str:
    title = re.sub(r"^[IVX]+\.\s+", "", title).strip()
    title = re.sub(r"^[A-Z]\.\s+", "", title).strip()
    return title


def clean_table_caption(raw: str) -> str:
    raw = raw.strip()
    match = re.match(r"^\*\*TABLE\s+([IVX]+)\.\*\*\s*(.*)$", raw)
    if match:
        return match.group(2).strip()
    return raw.strip("*").strip()


def table_to_latex(lines: list[str], caption: str, label: str) -> str:
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-+:?", c) for c in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""
    if not caption:
        header = tuple(rows[0])
        inferred = {
            ("Transform", "Detection Pattern", "Reversal"): "Obfuscation Decoding Transforms",
            ("Corpus Source", "Count"): "Benign Corpus Used for Layer 1 Anomaly Modeling",
            ("Component", "Specification"): "Experimental Hardware and Software Environment",
            ("Split", "Evaluated $n$", "Source", "Description"): "Final Holdout Benchmark Composition",
        }
        caption = inferred.get(header, "Summary Table")
    cols = len(rows[0])
    align = "l" + "c" * (cols - 1)
    env = "table*" if cols >= 6 or "Confidence" in caption or "Final Holdout" in caption else "table"
    width_cmd = r"\begin{adjustbox}{max width=\textwidth}" if env == "table*" else r"\begin{adjustbox}{max width=\columnwidth}"
    body = [
        rf"\begin{{{env}}}[!t]",
        r"\centering",
        rf"\caption{{{inline_latex(caption)}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        width_cmd,
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(inline_latex(c) for c in rows[0]) + r" \\",
        r"\midrule",
    ]
    for row in rows[1:]:
        body.append(" & ".join(inline_latex(c) for c in row) + r" \\")
    body.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            rf"\end{{{env}}}",
            "",
        ]
    )
    return "\n".join(body)


def code_block_to_latex(lines: list[str]) -> str:
    escaped = "\n".join(latex_escape(line) for line in lines)
    return "\n".join(
        [
            r"\begin{Verbatim}[fontsize=\scriptsize,breaklines=true]",
            escaped,
            r"\end{Verbatim}",
            "",
        ]
    )


def figure_latex(fig_no: int, caption: str) -> str:
    key = f"Fig. {fig_no}"
    path = FIGURE_FILES[key]
    env = "figure*" if fig_no in {2, 3, 5} else "figure"
    width = r"\textwidth" if env == "figure*" else r"\columnwidth"
    label = f"fig:{fig_no}"
    return "\n".join(
        [
            rf"\begin{{{env}}}[!t]",
            r"\centering",
            rf"\includegraphics[width={width}]{{{path}}}",
            rf"\caption{{{inline_latex(caption)}}}",
            rf"\label{{{label}}}",
            rf"\end{{{env}}}",
            "",
        ]
    )


def parse_references(lines: list[str]) -> list[str]:
    refs: list[str] = []
    current: list[str] = []
    for line in lines:
        if re.match(r"^\[\d+\]\s+", line):
            if current:
                refs.append(" ".join(current).strip())
            current = [line]
        elif current and line.strip():
            current.append(line.strip())
    if current:
        refs.append(" ".join(current).strip())
    return refs


def convert_body(lines: list[str]) -> str:
    out: list[str] = []
    pending_table_caption = ""
    pending_table_label = 1
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            out.append("")
            i += 1
            continue
        if stripped == "---":
            i += 1
            continue

        if stripped.startswith("**TABLE"):
            pending_table_caption = clean_table_caption(stripped)
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            label = f"tab:{pending_table_label}"
            out.append(table_to_latex(table_lines, pending_table_caption, label))
            pending_table_label += 1
            pending_table_caption = ""
            continue

        if stripped == "```":
            i += 1
            code_lines = []
            while i < len(lines) and lines[i].strip() != "```":
                code_lines.append(lines[i])
                i += 1
            i += 1
            out.append(code_block_to_latex(code_lines))
            continue

        fig_match = re.match(r"^\*Fig\.\s+([1-5])\.\s+(.*)\*$", stripped)
        if fig_match:
            out.append(figure_latex(int(fig_match.group(1)), fig_match.group(2)))
            i += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            title = strip_numbered_heading(heading.group(2))
            if level == 2:
                out.append(rf"\section{{{inline_latex(title)}}}")
            elif level == 3:
                out.append(rf"\subsection{{{inline_latex(title)}}}")
            else:
                out.append(rf"\subsubsection{{{inline_latex(title)}}}")
            i += 1
            continue

        bullet = re.match(r"^[-*]\s+(.*)$", stripped)
        number = re.match(r"^\d+\.\s+(.*)$", stripped)
        if bullet or number:
            ordered = bool(number)
            env = "enumerate" if ordered else "itemize"
            out.append(rf"\begin{{{env}}}")
            while i < len(lines):
                item = lines[i].strip()
                m = re.match(r"^\d+\.\s+(.*)$", item) if ordered else re.match(r"^[-*]\s+(.*)$", item)
                if not m:
                    break
                out.append(r"\item " + inline_latex(m.group(1)))
                i += 1
            out.append(rf"\end{{{env}}}")
            continue

        out.append(inline_latex(stripped) + "\n")
        if "Fig. 4 plots" in stripped:
            out.append(
                figure_latex(
                    4,
                    "Per-component mean and P95 latency on the final holdout benchmark.",
                )
            )
        if "Fig. 5 summarizes" in stripped:
            out.append(
                figure_latex(
                    5,
                    "Baseline and commercial comparison across the separate evaluation sets.",
                )
            )
        i += 1
    return "\n".join(out)


def main() -> None:
    lines = MD_PATH.read_text(encoding="utf-8").splitlines()
    title = lines[0].lstrip("# ").strip()

    abstract_start = lines.index("## Abstract") + 1
    keywords_idx = next(i for i, line in enumerate(lines) if line.startswith("**Keywords:**"))
    abstract_lines = [line for line in lines[abstract_start:keywords_idx] if line.strip() and line.strip() != "---"]
    abstract = "\n\n".join(inline_latex(line) for line in abstract_lines)
    keywords = lines[keywords_idx].replace("**Keywords:**", "").strip().rstrip(".")

    refs_heading = lines.index("## References")
    body_start = keywords_idx + 1
    while body_start < len(lines) and (not lines[body_start].strip() or lines[body_start].strip() == "---"):
        body_start += 1
    body = convert_body(lines[body_start:refs_heading])

    refs = parse_references(lines[refs_heading + 1 :])
    bib = [r"\begin{thebibliography}{20}"]
    for ref in refs:
        m = re.match(r"^\[(\d+)\]\s+(.*)$", ref)
        if not m:
            continue
        bib.append(rf"\bibitem{{ref{m.group(1)}}} {inline_latex(m.group(2))}")
    bib.append(r"\end{thebibliography}")

    tex = "\n".join(
        [
            r"\documentclass[conference]{IEEEtran}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{graphicx}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage{booktabs}",
            r"\usepackage{adjustbox}",
            r"\usepackage{array}",
            r"\usepackage{url}",
            r"\usepackage{fancyvrb}",
            r"\usepackage[hidelinks]{hyperref}",
            r"\IEEEoverridecommandlockouts",
            "",
            rf"\title{{{inline_latex(title)}}}",
            r"\author{\IEEEauthorblockN{Author Names Placeholder}\\",
            r"\IEEEauthorblockA{Institution Placeholder\\Email Placeholder}}",
            "",
            r"\begin{document}",
            r"\maketitle",
            "",
            r"\begin{abstract}",
            abstract,
            r"\end{abstract}",
            "",
            r"\begin{IEEEkeywords}",
            inline_latex(keywords),
            r"\end{IEEEkeywords}",
            "",
            r"\section*{Code and Artifact Availability}",
            r"The implementation and evaluation artifacts are available at \url{https://github.com/yomnafarag95/rag-defense-system}. "
            r"The paper repository is available at \url{https://github.com/yomnafarag95/IEEE_Paper}.",
            "",
            body,
            "\n".join(bib),
            "",
            r"\end{document}",
            "",
        ]
    )
    TEX_PATH.write_text(tex, encoding="utf-8")
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
