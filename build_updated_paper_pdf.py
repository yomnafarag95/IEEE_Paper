from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / "paper_draft.md"
OUTPUT = ROOT / "IEEE__Paper_updated_final_holdout.pdf"

FIGURES = {
    2: ROOT / "figures" / "fig2_roc_pr.png",
    3: ROOT / "figures" / "fig3_confusion_attribution.png",
    4: ROOT / "figures" / "fig4_latency.png",
    5: ROOT / "figures" / "fig5_method_comparison.png",
}


class PaperRenderer:
    def __init__(self, pdf: PdfPages) -> None:
        self.pdf = pdf
        self.fig = None
        self.col = 0
        self.y = 0.95
        self.page_no = 0
        self.in_code = False

    def _new_page(self) -> None:
        if self.fig is not None:
            self.fig.text(
                0.5,
                0.025,
                f"Page {self.page_no}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#555555",
            )
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.page_no += 1
        self.fig = plt.figure(figsize=(8.5, 11))
        self.fig.patch.set_facecolor("white")
        self.col = 0
        self.y = 0.95

    def _advance_column(self) -> None:
        if self.col == 0:
            self.col = 1
            self.y = 0.95
        else:
            self._new_page()

    def _ensure_page(self) -> None:
        if self.fig is None:
            self._new_page()

    def _put(self, text: str, *, size: float = 8.2, weight: str = "normal", mono: bool = False, color: str = "#111111") -> None:
        self._ensure_page()
        x = 0.06 if self.col == 0 else 0.53
        family = "DejaVu Sans Mono" if mono else "DejaVu Sans"
        line_h = (size / 72.0) / 11.0 * 1.45
        if self.y < 0.06 + line_h:
            self._advance_column()
            x = 0.06 if self.col == 0 else 0.53

        self.fig.text(x, self.y, text, ha="left", va="top", fontsize=size, fontweight=weight, family=family, color=color)
        self.y -= line_h

    def blank(self, amount: float = 0.012) -> None:
        self._ensure_page()
        self.y -= amount
        if self.y < 0.06:
            self._advance_column()

    def text(self, raw: str) -> None:
        line = clean_markdown(raw)
        if not line.strip():
            self.blank(0.008)
            return

        heading = re.match(r"^(#{1,4})\s+(.*)$", raw)
        if heading:
            level = len(heading.group(1))
            title = clean_markdown(heading.group(2))
            if level == 1:
                self._put(title, size=13, weight="bold")
            elif level == 2:
                self.blank(0.006)
                self._put(title.upper(), size=9.5, weight="bold")
            else:
                self._put(title, size=8.8, weight="bold")
            self.blank(0.004)
            return

        is_table = raw.lstrip().startswith("|")
        is_rule = set(raw.strip()) <= {"-"} and len(raw.strip()) >= 3
        if is_rule:
            self.blank(0.01)
            return

        if raw.strip() == "```":
            self.in_code = not self.in_code
            self.blank(0.004)
            return

        mono = self.in_code or is_table
        width = 54 if mono else 48
        size = 6.2 if mono else 7.8
        if raw.startswith("- "):
            line = "- " + clean_markdown(raw[2:])
            width = 46
        for part in textwrap.wrap(line, width=width, replace_whitespace=False, drop_whitespace=False) or [""]:
            self._put(part, size=size, mono=mono)

    def figure_page(self, fig_no: int, caption: str) -> None:
        path = FIGURES.get(fig_no)
        if path is None or not path.exists():
            self.text(caption)
            return

        if self.fig is not None:
            self._new_page()
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        img = plt.imread(path)
        ax = fig.add_axes([0.08, 0.18, 0.84, 0.68])
        ax.imshow(img)
        ax.axis("off")
        fig.text(0.08, 0.89, f"Figure {fig_no}", fontsize=12, fontweight="bold", ha="left", va="top")
        wrapped = textwrap.wrap(clean_markdown(caption), width=105)
        y = 0.13
        for line in wrapped:
            fig.text(0.08, y, line, fontsize=8.5, ha="left", va="top")
            y -= 0.018
        self.pdf.savefig(fig)
        plt.close(fig)
        self.fig = None

    def close(self) -> None:
        if self.fig is not None:
            self._new_page()


def clean_markdown(text: str) -> str:
    text = text.strip()
    text = text.strip("*")
    text = text.replace("**", "")
    text = text.replace("`", "")
    text = text.replace("$", "")
    text = text.replace("\\", "")
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def build() -> None:
    content = MANUSCRIPT.read_text(encoding="utf-8").splitlines()
    with PdfPages(OUTPUT) as pdf:
        renderer = PaperRenderer(pdf)
        for raw in content:
            caption_match = re.match(r"^\*Fig\.\s+([2-5])\.\s+(.*)\*$", raw.strip())
            if caption_match:
                renderer.figure_page(int(caption_match.group(1)), raw)
                continue
            renderer.text(raw)
        renderer.close()

    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
