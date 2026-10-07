#!/usr/bin/env python3
"""Build the final Team 18 DATA266 Lab 1 report from committed evidence.

The PDF is intentionally generated from the repository artifacts so that every
reported number can be traced to a CSV, JSON, notebook output, log, or figure.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    LongTable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
OUTPUT = REPORT_DIR / "DATA266_Lab1_Report_Team_18.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2166AC")
TEAL = colors.HexColor("#167D7F")
GREEN = colors.HexColor("#247A4B")
PALE_GREEN = colors.HexColor("#EAF6EF")
PALE_BLUE = colors.HexColor("#EAF2FA")
PALE_AMBER = colors.HexColor("#FFF5DD")
AMBER = colors.HexColor("#B36B00")
LIGHT = colors.HexColor("#F3F6F8")
MID = colors.HexColor("#D7E0E7")
DARK_GREY = colors.HexColor("#3D4A55")
MUTED = colors.HexColor("#657681")


def read_csv(rel: str) -> list[dict[str, str]]:
    with (ROOT / rel).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def compact(text: str, limit: int = 180) -> str:
    text = " ".join(text.replace("\\n", " ").replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def num(value: object, digits: int = 4, missing: str = "-") -> str:
    if value is None or value == "":
        return missing
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return missing
    return f"{x:.{digits}f}"


class ReportDocTemplate(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style = flowable.style.name
            if style in {"H1", "H2"}:
                level = 0 if style == "H1" else 1
                if level == 1 and (self.page == 1 or not getattr(self, "_outline_started", False)):
                    return
                if level == 0:
                    self._outline_started = True
                text = flowable.getPlainText()
                key = f"heading-{level}-{self.page}-{abs(hash(text))}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=level, closed=False)
                self.notify("TOCEntry", (level, text, self.page, key))


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=27,
        leading=32,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=14,
    )
)
styles.add(
    ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=TEAL,
        spaceAfter=12,
    )
)
styles.add(
    ParagraphStyle(
        name="H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=NAVY,
        spaceBefore=6,
        spaceAfter=10,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=BLUE,
        spaceBefore=9,
        spaceAfter=6,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13,
        textColor=DARK_GREY,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.7,
        leading=10.2,
        textColor=DARK_GREY,
        spaceAfter=3,
    )
)
styles.add(
    ParagraphStyle(
        name="Tiny",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=6.7,
        leading=8.4,
        textColor=DARK_GREY,
    )
)
styles.add(
    ParagraphStyle(
        name="Caption",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=9.5,
        alignment=TA_CENTER,
        textColor=MUTED,
        spaceBefore=3,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="CodePath",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=6.8,
        leading=8.5,
        textColor=MUTED,
        spaceAfter=3,
    )
)
styles.add(
    ParagraphStyle(
        name="Callout",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12.5,
        textColor=DARK_GREY,
    )
)
styles.add(
    ParagraphStyle(
        name="TOCHeading",
        parent=styles["Heading2"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=DARK_GREY,
        leftIndent=0,
        firstLineIndent=0,
        spaceBefore=2,
    )
)


def P(text: str, style: str = "BodyCustom") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str, level: int = 0) -> Paragraph:
    style = ParagraphStyle(
        name=f"Bullet{level}",
        parent=styles["BodyCustom"],
        leftIndent=12 + 12 * level,
        firstLineIndent=-7,
        bulletIndent=3 + 12 * level,
        spaceAfter=3,
    )
    return Paragraph(esc(text), style, bulletText="-")


def callout(title: str, text: str, kind: str = "blue") -> Table:
    if kind == "green":
        bg, edge = PALE_GREEN, GREEN
    elif kind == "amber":
        bg, edge = PALE_AMBER, AMBER
    else:
        bg, edge = PALE_BLUE, BLUE
    content = P(f"<b>{esc(title)}</b><br/>{text}", "Callout")
    table = Table([[content]], colWidths=[6.75 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 1.1, edge),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def cell(value: object, tiny: bool = False, bold: bool = False) -> Paragraph:
    txt = esc(value)
    if bold:
        txt = f"<b>{txt}</b>"
    return P(txt, "Tiny" if tiny else "Small")


def styled_table(
    rows: list[list[object]],
    widths: list[float],
    *,
    font_size: float = 7.2,
    repeat_rows: int = 1,
    landscape_mode: bool = False,
    alignments: dict[int, str] | None = None,
    padding: float = 4,
) -> LongTable:
    data: list[list[Paragraph]] = []
    for r, row in enumerate(rows):
        converted = []
        for value in row:
            if isinstance(value, Paragraph):
                converted.append(value)
            else:
                converted.append(
                    Paragraph(
                        (f"<b>{esc(value)}</b>" if r == 0 else esc(value)),
                        ParagraphStyle(
                            name=f"table-{font_size}-{r}",
                            parent=styles["Tiny"],
                            fontName="Helvetica-Bold" if r == 0 else "Helvetica",
                            fontSize=font_size,
                            leading=font_size + 2,
                            textColor=colors.white if r == 0 else DARK_GREY,
                            alignment=TA_LEFT,
                        ),
                    )
                )
        data.append(converted)
    table = LongTable(data, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, MID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), padding),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
    ]
    for r in range(1, len(rows)):
        if r % 2 == 0:
            commands.append(("BACKGROUND", (0, r), (-1, r), LIGHT))
    if alignments:
        for col, alignment in alignments.items():
            commands.append(("ALIGN", (col, 1), (col, -1), alignment))
    table.setStyle(TableStyle(commands))
    return table


IMAGE_CACHE = Path(tempfile.mkdtemp(prefix="lab1_report_images_"))
PRINT_DPI = 220


def scaled_image(path: Path, max_w: float, max_h: float) -> Image:
    with PILImage.open(path) as img:
        width, height = img.size
        scale = min(max_w / width, max_h / height)
        # Downsample to print resolution so the PDF stays small enough to push.
        target_w = int(width * scale / 72 * PRINT_DPI)
        if target_w < width:
            target_h = max(1, round(height * target_w / width))
            cached = IMAGE_CACHE / f"{abs(hash((str(path), target_w)))}.jpg"
            if not cached.exists():
                img.convert("RGB").resize((target_w, target_h), PILImage.LANCZOS).save(cached, quality=88)
            path = cached
    return Image(str(path), width=width * scale, height=height * scale)


def figure(rel: str, caption: str, max_w: float = 6.75 * inch, max_h: float = 7.0 * inch):
    path = ROOT / rel
    return [
        scaled_image(path, max_w, max_h),
        P(f"<b>Figure.</b> {esc(caption)}", "Caption"),
        P(esc(rel), "CodePath"),
    ]


def two_figures(rel_a: str, cap_a: str, rel_b: str, cap_b: str) -> Table:
    max_w = 3.23 * inch
    img_a = scaled_image(ROOT / rel_a, max_w, 3.25 * inch)
    img_b = scaled_image(ROOT / rel_b, max_w, 3.25 * inch)
    table = Table(
        [
            [img_a, img_b],
            [P(esc(cap_a), "Caption"), P(esc(cap_b), "Caption")],
        ],
        colWidths=[3.35 * inch, 3.35 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def page_header_footer(canvas, doc):
    width, height = canvas._pagesize
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(MID)
        canvas.setLineWidth(0.5)
        canvas.line(0.62 * inch, height - 0.46 * inch, width - 0.62 * inch, height - 0.46 * inch)
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.62 * inch, height - 0.35 * inch, "DATA266 Lab 1 | Team 18")
        canvas.drawRightString(width - 0.62 * inch, height - 0.35 * inch, "Joint technical report")
        canvas.line(0.62 * inch, 0.42 * inch, width - 0.62 * inch, 0.42 * inch)
        canvas.drawCentredString(width / 2, 0.25 * inch, f"Page {doc.page}")
    canvas.restoreState()


def switch_landscape(story: list):
    story.extend([NextPageTemplate("landscape"), PageBreak()])


def switch_portrait(story: list):
    story.extend([NextPageTemplate("portrait"), PageBreak()])


def task1_rows() -> tuple[dict[str, str], dict[str, str]]:
    c = read_csv("task1_llm/member_chaitanya/metrics_report.csv")[0]
    a_rows = read_csv("task1_llm/member_aswin/metrics_report.csv")
    a = next(row for row in a_rows if row["run_id"] == "full_run01")
    return c, a


def task2_rows() -> list[tuple[str, dict[str, str]]]:
    c_rows = [
        row
        for row in read_csv("task2_sentiment/member_chaitanya/metrics_report.csv")
        if row["eval_set"] == "test5k" and not row["model"].startswith("Reference")
    ]
    a_rows = read_csv("task2_sentiment/member_aswin/metrics_report.csv")
    labels = [
        ("C: fastText bigram", c_rows[0]),
        ("C: HAN", c_rows[1]),
        ("C: Transformer", c_rows[2]),
        ("A: mean-pool FFN", a_rows[0]),
        ("A: BiGRU", a_rows[1]),
        ("A: TextCNN", a_rows[2]),
    ]
    return labels


def task3_local_rows():
    c = [row for row in read_csv("task3_gan/member_chaitanya/full_metrics_report.csv") if row["run_id"] == "full_run18"]
    a = read_csv("task3_gan/member_aswin/full_metrics_report.csv")
    return c, a


def add_evidence_table(story: list, rows: list[tuple[str, str]]):
    table_rows = [["Evidence", "Repository path / identifier"]] + [[name, path] for name, path in rows]
    story.append(styled_table(table_rows, [1.35 * inch, 5.35 * inch], font_size=6.9))


def build_story() -> list:
    story: list = []

    # Title page
    story.append(Spacer(1, 0.55 * inch))
    story.append(P("DATA266 Lab 1", "TitleCustom"))
    story.append(P("LLM pretraining, sentiment classification, and CycleGAN style transfer", "Subtitle"))
    story.append(Spacer(1, 0.18 * inch))
    title_box = Table(
        [
            [P("<b>Team</b>", "Small"), P("18", "Small")],
            [P("<b>Members</b>", "Small"), P("Aswin John and Chaitanya Uppalapati", "Small")],
            [P("<b>Repository</b>", "Small"), P('<link href="https://github.com/ChaitanyaUppalapati/Lab_Pair_18" color="#2166AC">github.com/ChaitanyaUppalapati/Lab_Pair_18</link>', "Small")],
            [P("<b>Report date</b>", "Small"), P("October 6, 2026", "Small")],
        ],
        colWidths=[1.25 * inch, 5.25 * inch],
    )
    title_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.8, MID),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, MID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(title_box)
    story.append(Spacer(1, 0.35 * inch))
    story.append(
        callout(
            "Team-selected results",
            "Task 1: Chaitanya's 10.8M-parameter GPT (validation CE 0.5097). "
            "Task 2: Aswin's TextCNN (92.52% accuracy, 2.59M parameters), selected as the lead Task 2 model. "
            "Task 3: Chaitanya's final CycleGAN submission (FID 96.0061, MiFID 0.3998, public score -48.2029, team rank 15th).",
            "green",
        )
    )
    story.append(Spacer(1, 0.25 * inch))
    story.append(P("Purpose", "H2"))
    story.append(
        P(
            "This joint report compares every independently trained member model, selects the strongest team result for each task, "
            "and traces all claims to repository evidence. It combines quantitative evaluation with loss curves, qualitative outputs, "
            "failure cases, robustness analysis, hardware disclosure, and reproducibility notes."
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(P("Integrity note", "H2"))
    story.append(
        P(
            "No metric, human rating, private leaderboard score, or rank is inferred when supporting evidence is absent. "
            "The private leaderboard fields are therefore left open until the competition closes."
        )
    )
    story.append(PageBreak())

    # Ownership and contents
    story.append(P("Team ownership statement", "H1"))
    story.append(
        P(
            "Aswin and Chaitanya each maintained named folders and independently ran all three tasks. Aswin trained the compact "
            "character GPT on an A100, the three Yelp classifiers on Apple M4/MPS, and the V2 CycleGAN on an RTX 4090, and preserved "
            "the executed notebooks, checkpoints, predictions, metrics, and plots. Chaitanya designed and ran the larger GPT, the fastText/HAN/"
            "Transformer sentiment lineup, and the iterative CycleGAN experiments on an RTX 4090. Implementation and debugging used AI "
            "assistance: Chaitanya's code was written with an AI assistant from Chaitanya's design decisions, and Task 3 decisions from run 5 "
            "onward are marked as AI-assisted in DESIGN_LOG.md. Aswin's code was written with AI coding assistants (OpenAI Codex and Claude) "
            "from Aswin's design decisions; Aswin chose the architectures and hyperparameters, ran the training, and reviewed the outputs. "
            "This report was compiled with AI assistance from the repository evidence. Both members rated the 30-sample Task 3 human audit. Both members own "
            "the submitted artifacts and remain responsible for explaining their individual choices and results. This report was synthesized "
            "from the recorded evidence and should be reviewed by both members before submission."
        )
    )
    story.append(P("Evaluation and selection policy", "H2"))
    story.append(bullet("Use the shared 5,000-review test split for cross-member Task 2 comparisons."))
    story.append(bullet("Use the instructor's first-300-images evaluation notebook for the official Task 3 FID/MiFID comparison."))
    story.append(bullet("Treat protocol differences as confounds, and do not call a difference architectural when preprocessing or data exposure also differs."))
    story.append(bullet("Select by the task's primary held-out or official score; use efficiency and qualitative behavior as secondary criteria."))
    story.append(P("Contents", "H2"))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(name="TOC1", fontName="Helvetica-Bold", fontSize=9, leading=13, leftIndent=0, textColor=NAVY),
        ParagraphStyle(name="TOC2", fontName="Helvetica", fontSize=8, leading=11, leftIndent=14, textColor=DARK_GREY),
    ]
    story.append(toc)
    story.append(PageBreak())

    # Task 1
    c1, a1 = task1_rows()
    story.append(P("1. Task 1 - Character-level GPT from scratch", "H1"))
    story.append(
        P(
            "Both models implement causal decoder-only Transformers without prebuilt Transformer or attention modules. The models differ "
            "substantially in capacity, context length, cleaning, and how much text is exposed per epoch, so the comparison measures complete "
            "pipelines rather than a single controlled ablation."
        )
    )
    story.append(P("1.1 Architecture, data, and training", "H2"))
    t1_design = [
        ["Item", "Chaitanya full_run01", "Aswin full_run01"],
        ["Data", "100K train / 10K validation stories; cleaned, exact-deduplicated; sliding windows, stride 128", "100K train / 10K validation stories; one seeded crop or padded sequence per story"],
        ["Vocabulary", "85 characters incl. EOS/UNK; mojibake stories removed", "111 entries incl. BOS/EOS/PAD/UNK; raw character coverage retained"],
        ["Architecture", "6 pre-LN blocks; d_model 384; 6 heads x 64; FFN 1536; learned positions", "5 pre-LN blocks; d_model 240; 6 heads x 40; FFN 960; learned positions"],
        ["Context / dropout", "256 / 0.10", "160 / 0.12"],
        ["Optimizer", "AdamW, lr 1e-3, wd 0.1, betas 0.9/0.99", "AdamW, lr 2.5e-4, wd 0.1"],
        ["Schedule", "1,000-step warm-up; cosine to 1e-4", "5% warm-up; cosine to 10% of peak"],
        ["Batch / epochs", "64 / 10", "48 / 12"],
        ["Hardware", "RTX 4090, fp32", "Google Colab Pro A100-SXM4-40GB"],
    ]
    story.append(styled_table(t1_design, [1.25 * inch, 2.7 * inch, 2.7 * inch], font_size=7.2))
    story.append(P("1.2 Complete comparison metrics", "H2"))
    t1_metrics = [
        ["Metric", "Chaitanya", "Aswin"],
        ["Training CE loss", num(c1["train_ce_loss"]), num(a1["train_ce_loss"])],
        ["Validation CE loss", num(c1["val_ce_loss"]), num(a1["val_ce_loss"])],
        ["Perplexity", num(c1["perplexity"]), num(a1["perplexity"])],
        ["Bits per character", num(c1["bits_per_char"]), num(a1["bits_per_char"])],
        ["Generalization gap", num(c1["generalization_gap"]), num(a1["generalization_gap"])],
        ["Top-1 next-char accuracy", num(c1["top1_next_char_acc"]), num(a1["top1_next_char_acc"])],
        ["Distinct-1 / 2 / 3", f"{num(c1['distinct_1'])} / {num(c1['distinct_2'])} / {num(c1['distinct_3'])}", f"{num(a1['distinct_1'])} / {num(a1['distinct_2'])} / {num(a1['distinct_3'])}"],
        ["Repeated 4-gram rate", num(c1["repeated_4gram_rate"]), num(a1["repeated_4gram_rate"])],
        ["Gradient norm max / mean", f"{num(c1['max_grad_norm'], 3)} / {num(c1['mean_grad_norm'], 3)}", f"{num(a1['max_grad_norm'], 3)} / {num(a1['mean_grad_norm'], 3)}"],
        ["Loss spikes / NaNs", f"{c1['loss_spikes']} / {c1['nan_count']}", f"{a1['loss_spikes']} / {a1['nan_count']}"],
        ["Parameters", f"{int(float(c1['param_count'])):,}", f"{int(float(a1['param_count'])):,}"],
        ["Train tokens/s", f"{float(c1['train_tokens_per_sec']):,.0f}", f"{float(a1['train_tokens_per_sec']):,.0f}"],
        ["Generation tokens/s", num(c1["gen_tokens_per_sec"], 1), num(a1["gen_tokens_per_sec"], 1)],
        ["Peak memory", f"{float(c1['peak_memory_mb']):,.0f} MB", f"{float(a1['peak_memory_mb']):,.0f} MB"],
        ["Training time", f"{float(c1['total_train_time_s']):,.0f} s", f"{float(a1['total_train_time_s']):,.0f} s"],
    ]
    story.append(styled_table(t1_metrics, [2.2 * inch, 2.2 * inch, 2.2 * inch], font_size=7.3))
    story.append(Spacer(1, 6))
    story.append(
        callout(
            "Team selection: Chaitanya full_run01",
            "It has the lower validation loss (0.5097 vs 0.7582), lower perplexity (1.665 vs 2.134), and higher next-character accuracy "
            "(83.58% vs 76.01%). Aswin's model is the efficiency winner: it is about one third the size, trains about 10.3x faster in wall-clock "
            "time, uses about 4.5x less peak memory, and generates faster. Distinct-n values are not directly comparable because one pipeline "
            "uses word n-grams and the other character n-grams.",
            "green",
        )
    )
    story.append(P("1.3 Training evidence", "H2"))
    story.append(
        two_figures(
            "task1_llm/member_chaitanya/outputs/full_run01/loss_curves.png",
            "Chaitanya: step-level and epoch-level curves. Validation loss falls through epoch 10 with a small positive gap.",
            "task1_llm/member_aswin/outputs/plots/full_run01_loss_curves.png",
            "Aswin: loss falls through epoch 12. The lower validation curve reflects dropout being disabled during validation, not leakage.",
        )
    )
    story.append(P("1.4 Joint analysis", "H2"))
    story.append(
        P(
            "The stronger GPT benefits from roughly 3x more parameters, 60% longer context, aggressive cleaning/deduplication, and sliding-window "
            "exposure that provides roughly ten times more target characters per epoch. Its zero-spike trace and 0.0255 generalization gap show "
            "stable underfitting: validation loss was still decreasing, so additional epochs or mixed precision would likely improve it. Aswin's "
            "compact run demonstrates excellent throughput and memory efficiency, but one crop per story and a 160-character context weaken story "
            "coherence. The negative reported gap in that run mixes dropout-on training loss with dropout-off validation loss and should not be "
            "interpreted as genuine validation superiority. A controlled follow-up would standardize the split, vocabulary, metric definitions, and "
            "number of target characters seen, then ablate context length and model size separately."
        )
    )
    story.append(P("1.5 Failure cases with actual generated text", "H2"))
    fail_rows = [
        ["Member / case", "Generated excerpt", "Failure and diagnosis"],
        ["Chaitanya 1", '"The dog and the box became best friends. They played together every day." repeats twice.', "Greedy repetition loop; a frequent template becomes self-reinforcing."],
        ["Chaitanya 2", '"an x-ray of te-ta-ta--cars ... clean her tea-flashers"', "Temperature-1.0 character sampling creates invented words and semantic drift."],
        ["Chaitanya 3", '"learnt an important lesson - that woggly river my work tasks"', "A familiar moral template begins correctly but long-range syntax collapses."],
        ["Aswin 1", '"The bird was so happy that he had been so much fun to see the bird" repeats to the cap.', "Greedy repetition, amplified by short context and single-crop training."],
        ["Aswin 2", '"a bugs ... a cloth their basket ... mojibake quotation marks"', "High-temperature output exposes retained encoding artifacts and grammar errors."],
        ["Aswin 3", '"little swept and smell ... played in the sleep ... became to healthy"', "Words are locally plausible but the event sequence and argument structure fail."],
    ]
    story.append(styled_table(fail_rows, [0.9 * inch, 2.7 * inch, 3.05 * inch], font_size=6.9))
    story.append(
        P(
            "The shared pattern is a mismatch between local character fluency and story-level planning. Lower-temperature or nucleus sampling, "
            "repetition blocking, cleaner Unicode normalization, longer context, and multiple windows per story are directly testable remedies."
        )
    )
    story.append(P("1.6 Evidence map", "H2"))
    add_evidence_table(
        story,
        [
            ("Chaitanya metrics", "task1_llm/member_chaitanya/metrics_report.csv"),
            ("Chaitanya checkpoint", "task1_llm/member_chaitanya/checkpoints/full_run01/final.pt; hash in reproducibility manifest"),
            ("Chaitanya raw log", "reproducibility/raw_logs/task1_llm/chaitanya/full_run01.log"),
            ("Aswin metrics", "task1_llm/member_aswin/metrics_report.csv (full_run01 row)"),
            ("Aswin executed notebook", "task1_llm/member_aswin/src/task1_colab_all_in_one.ipynb"),
            ("Aswin samples", "task1_llm/member_aswin/outputs/generated_text/full_run01_samples.jsonl"),
        ],
    )
    story.append(PageBreak())

    # Task 2
    rows2 = task2_rows()
    story.append(P("2. Task 2 - Yelp polarity sentiment classification", "H1"))
    story.append(
        P(
            "All six counted models use embeddings learned from scratch and are evaluated on the same stratified 5,000-review test set "
            "(2,500 positive, 2,500 negative). Chaitanya reproduced Aswin's row indices and verified texts and labels, enabling paired tests. "
            "Preprocessing still differs, so cross-member gaps combine preprocessing and architecture effects."
        )
    )
    story.append(P("2.1 Model lineup", "H2"))
    design2 = [
        ["Model", "Architecture", "Parameters", "Key training choices"],
        ["C: fastText bigram", "Mean unigram + hashed-bigram embeddings -> linear", "19.29M", "128-d; AdamW 3e-3; batch 64; early stop; dropout 0.3"],
        ["C: HAN", "Word BiLSTM + attention -> sentence BiLSTM + attention", "2.75M", "128-d; AdamW 1e-3; batch 64; dropout 0.3; clip 1.0"],
        ["C: Transformer", "2-layer, 4-head, width-128 hand-written encoder; masked mean pool", "2.94M", "AdamW 5e-4; 5% warm-up; dropout 0.1/0.2; clip 1.0"],
        ["A: mean-pool FFN", "Mean pool -> Dense 128 -> ReLU -> dropout -> logit", "2.42M", "128-d; AdamW 1e-3; wd 1e-4; batch 64; <=5 epochs"],
        ["A: BiGRU", "1-layer BiGRU 128/dir -> max+mean pool -> Dense 128", "2.67M", "AdamW 5e-4; dropout 0.4; batch 64; <=5 epochs"],
        ["A: TextCNN", "Conv1D kernels 3/4/5 x 100 -> max pool -> Dense 128", "2.59M", "AdamW 1e-3; dropout 0.4; batch 64; <=5 epochs"],
    ]
    story.append(styled_table(design2, [1.12 * inch, 2.65 * inch, 0.7 * inch, 2.2 * inch], font_size=6.8))
    core = [["Model", "Acc.", "Macro P/R/F1", "Micro F1", "Weighted F1", "TN/FP/FN/TP", "ROC / PR", "MCC"]]
    for label, row in rows2:
        core.append(
            [
                label,
                num(row["accuracy"]),
                f"{num(row['precision_macro'])} / {num(row['recall_macro'])} / {num(row['f1_macro'])}",
                num(row["f1_micro"]),
                num(row["f1_weighted"]),
                f"{row['tn']}/{row['fp']}/{row['fn']}/{row['tp']}",
                f"{num(row['roc_auc'])} / {num(row['pr_auc'])}",
                num(row["mcc"]),
            ]
        )
    switch_landscape(story)
    story.append(P("2.2 Shared test-set metrics", "H1"))
    story.append(styled_table(core, [1.1 * inch, 0.55 * inch, 1.4 * inch, 0.65 * inch, 0.72 * inch, 1.1 * inch, 0.95 * inch, 0.62 * inch], font_size=6.7, landscape_mode=True))
    stats_rows = [["Model", "Acc. 95% CI", "Macro-F1 95% CI", "MCC 95% CI", "Brier", "ECE", "McNemar vs own baseline", "Params", "Time", "Examples/s", "Peak MB"]]
    for label, row in rows2:
        if label.endswith("bigram") or label.endswith("FFN"):
            mc = "reference"
        else:
            p = row.get("mcnemar_vs_baseline_p", "")
            s = row.get("mcnemar_vs_baseline_stat", "")
            mc = f"stat {num(s, 3)}, p {num(p, 4)}" if p else "-"
        stats_rows.append(
            [
                label,
                f"[{num(row['acc_ci_low'])}, {num(row['acc_ci_high'])}]",
                f"[{num(row['f1_macro_ci_low'])}, {num(row['f1_macro_ci_high'])}]",
                f"[{num(row['mcc_ci_low'])}, {num(row['mcc_ci_high'])}]",
                num(row["brier"]),
                num(row["ece"]),
                mc,
                f"{int(float(row['param_count']))/1e6:.2f}M",
                f"{float(row['train_time_s']):.1f}s",
                f"{float(row['examples_per_sec']):,.0f}",
                f"{float(row['peak_memory_mb']):,.0f}",
            ]
        )
    story.append(Spacer(1, 8))
    story.append(styled_table(stats_rows, [1.05 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.52 * inch, 0.48 * inch, 1.15 * inch, 0.55 * inch, 0.52 * inch, 0.65 * inch, 0.55 * inch], font_size=6.1, landscape_mode=True))
    story.append(Spacer(1, 8))
    story.append(
        callout(
            "Team selection: Aswin TextCNN",
            "Aswin's TextCNN is the lead Task 2 model in this report: it is the strongest of Aswin's three runs (0.9252 accuracy, 0.9252 macro-F1, "
            "0.9800 ROC-AUC) and uses only 2.59M parameters. Chaitanya's fastText-bigram baseline has a slightly higher point estimate (0.9294), "
            "but the paired difference is not significant after Holm correction (adjusted p = 0.9012). The TextCNN therefore gives the preferred "
            "accuracy/compactness trade-off while keeping Aswin's executed notebook and artifacts as the primary Task 2 evidence.",
            "green",
        )
    )
    story.append(P("2.3 Slice robustness", "H2"))
    slices = read_csv("task2_sentiment/member_chaitanya/outputs/slice_metrics.csv")
    model_map = {
        "Baseline (fastText bigram)": "C: fastText",
        "Exp 1 (HAN)": "C: HAN",
        "Exp 2 (Transformer)": "C: Transformer",
        "Aswin baseline (mean-pool FFN)": "A: FFN",
        "Aswin exp 1 (BiGRU)": "A: BiGRU",
        "Aswin exp 2 (TextCNN)": "A: TextCNN",
    }
    slice_order = [
        "A: Short Reviews (<50 words)",
        "A: Medium Reviews (50-150 words)",
        "A: Long Reviews (>150 words)",
        "A: Negation-Bearing",
        "A: Strong Sentiment Cues",
        "A: Mixed Sentiment",
        "A: High Unknown Token (>5% UNK)",
    ]
    slice_labels = ["Short", "Medium", "Long", "Negation", "Strong cues", "Mixed", ">5% UNK"]
    slice_table = [["Model"] + [f"{name} F1/err" for name in slice_labels]]
    for model, short_name in model_map.items():
        vals = []
        for slice_name in slice_order:
            match = next((r for r in slices if r["eval_set"] == "test5k" and r["model"] == model and r["slice"] == slice_name), None)
            vals.append(f"{num(match['macro_f1'], 3)}/{num(match['error_rate'], 3)}" if match else "-")
        slice_table.append([short_name] + vals)
    story.append(styled_table(slice_table, [0.95 * inch] + [0.91 * inch] * 7, font_size=6.3, landscape_mode=True))
    story.append(
        P(
            "Cells show macro-F1 / error rate. Explicit sentiment cues are easiest. Mixed sentiment and high unknown-token rates expose the main "
            "weaknesses. TextCNN is the strongest Aswin model on medium, negation, strong-cue, and mixed slices; the local n-gram bias aligns with "
            "the bigram baseline's overall success."
        )
    )
    switch_portrait(story)
    story.append(P("2.4 Aswin Task 2 evidence", "H1"))
    story.append(
        two_figures(
            "task2_sentiment/member_aswin/outputs/plots/experimental_2_roc_curve.png",
            "Aswin TextCNN ROC curve on the shared test split (ROC-AUC 0.9800).",
            "task2_sentiment/member_aswin/outputs/plots/experimental_2_pr_curve.png",
            "Aswin TextCNN precision-recall curve (PR-AUC 0.9804).",
        )
    )
    story.append(
        two_figures(
            "task2_sentiment/member_aswin/outputs/confusion_matrices/experimental_2_confusion_matrix.png",
            "Aswin TextCNN confusion matrix: TN 2,290; FP 210; FN 164; TP 2,336.",
            "task2_sentiment/member_aswin/outputs/calibration/experimental_2_reliability_diagram.png",
            "Aswin TextCNN reliability diagram (Brier 0.0556; ECE 0.0177 in the member metrics file).",
        )
    )
    story.append(
        two_figures(
            "task2_sentiment/member_aswin/outputs/plots/experimental_2_training_curves.png",
            "Aswin TextCNN training curves and validation behavior.",
            "task2_sentiment/member_aswin/outputs/plots/slice_robustness_comparison.png",
            "Aswin's seven-slice robustness comparison across the baseline, BiGRU, and TextCNN.",
        )
    )
    story.append(P("2.5 Joint error analysis", "H2"))
    errors = [
        ["Member / category", "Actual review excerpt", "What failed / testable response"],
        ["C: confident FP", '"Average Japanese food at amazing Japanese food prices."', "Sarcastic use of 'amazing'; add sentiment-incongruity or sarcasm features."],
        ["C: confident FN", '"Sunday buffet for $13 all you can eat and drink."', "Positive label with no explicit sentiment cue; combine text with weak rating/context signals if allowed."],
        ["C: negation", '"despite still not digging their ordering process, their food is just too good..."', "Scope and double-negative structure; add last-sentence/verdict pooling and test on contrast slices."],
        ["C: mixed", '"Very good staff ... The food is wretched ..."', "Mean pooling sums conflicting cues; learn clause-level weighting."],
        ["A: confident FP", '"Laser quest is a fun experience ... adults ... creep me out."', "Mixed review dominated by positive vocabulary; add clause/position weighting."],
        ["A: confident FN", '"Owner wrote me and asked me to actually give his business another chance..."', "Subtle positive intent has no strong sentiment cue; the full audit proposes stronger discourse/context modeling."],
        ["A: near threshold", "Long reviews with opposing clauses and late verdicts", "Sequence truncation and weak conclusion weighting; hierarchical or last-sentence pooling is testable."],
        ["A: slice-specific", "Reviews with negation or mixed sentiment", "TextCNN local features help, but cannot always resolve which clause determines the label."],
    ]
    story.append(styled_table(errors, [1.15 * inch, 2.55 * inch, 2.95 * inch], font_size=6.8))
    story.append(
        P(
            "Aswin's 20-case audit is the primary Task 2 error analysis; Chaitanya's audit is retained for the required team comparison. Across "
            "both audits the dominant errors are mixed reviews, sarcasm, mild/factual sentiment, negation scope, and occasional label "
            "ambiguity. These are reasoning and discourse failures rather than ordinary vocabulary misses. Appendix A includes all 40 reviewed "
            "cases in compact form; the repository CSVs retain the complete text and proposed fixes."
        )
    )
    story.append(P("2.6 Joint analysis and next steps", "H2"))
    story.append(
        P(
            "Aswin's TextCNN is the selected model because its convolutional filters preserve local 3-5 token expressions such as 'not good' "
            "while remaining compact. More generally, local phrase bias wins at this data scale: hashed bigrams and TextCNN filters preserve "
            "expressions that mean "
            "pooling loses, while the scratch Transformer lacks enough data to learn equivalent locality and is the least calibrated. The HAN's "
            "hierarchy does not improve mixed-review performance because many polarity reversals occur within a sentence. A fair next experiment "
            "would standardize preprocessing, train all models over three seeds, add conclusion-aware pooling, and measure paired McNemar changes "
            "on contrast, mixed, and unknown-token slices. The uncounted TF-IDF+logistic-regression reference reaches 0.935 accuracy, showing that "
            "strong sparse n-grams remain difficult to beat with only 20K training reviews."
        )
    )
    story.append(P("2.7 Evidence map", "H2"))
    add_evidence_table(
        story,
        [
            ("Lead executed notebook", "task2_sentiment/member_aswin/src/task2_sentiment_aswin.ipynb"),
            ("Aswin metrics", "task2_sentiment/member_aswin/metrics_report.csv"),
            ("Aswin error review", "task2_sentiment/member_aswin/outputs/error_analysis/error_analysis.csv"),
            ("Aswin plots", "task2_sentiment/member_aswin/outputs/plots/, outputs/confusion_matrices/, outputs/calibration/"),
            ("Unified comparison", "task2_sentiment/member_chaitanya/outputs/team_comparison.csv"),
            ("Paired tests", "task2_sentiment/member_chaitanya/outputs/team_mcnemar.csv"),
            ("Slice table", "task2_sentiment/member_chaitanya/outputs/slice_metrics.csv"),
            ("Chaitanya metrics", "task2_sentiment/member_chaitanya/metrics_report.csv"),
            ("Chaitanya error review", "task2_sentiment/member_chaitanya/outputs/error_review/baseline_candidates.csv"),
        ],
    )
    story.append(PageBreak())

    # Task 3
    c3, a3 = task3_local_rows()
    c3_by_dir = {r["direction"]: r for r in c3}
    a3_by_dir = {r["direction"]: r for r in a3}
    with (ROOT / "task3_gan/member_chaitanya/outputs/official_eval.json").open(encoding="utf-8") as stream:
        c3_official = json.load(stream)
    aswin_fid = 105.1144417304836
    aswin_mifid = 0.4136924761280517
    aswin_score = -(aswin_fid + aswin_mifid) / 2
    story.append(P("3. Task 3 - CycleGAN Monet/photo style transfer", "H1"))
    story.append(
        P(
            "A denotes Monet and B denotes photo. Every generated image used for evaluation is direct output from the members' own CycleGAN "
            "generators. The official comparison uses the instructor notebook on the first 300 sorted images in each real and generated folder; "
            "the score is -(FID + MiFID)/2, so higher (less negative) is better."
        )
    )
    story.append(P("3.1 Architecture and training comparison", "H2"))
    t3_design = [
        ["Item", "Chaitanya selected model", "Aswin V2"],
        ["Generators", "ResNet-9, 64 filters, InstanceNorm, reflection padding, nearest-neighbor resize-conv", "ResNet-9, 64 filters, InstanceNorm, reflection padding, nearest-neighbor resize-conv"],
        ["Discriminators", "A2B lineage single-scale; B2A specialization uses two-scale 70x70 PatchGAN", "Single-scale 70x70 PatchGAN with InstanceNorm (no spectral normalization)"],
        ["Objective", "LSGAN; lambda_cycle 2; lambda_identity 0 in selected lineage", "LSGAN; lambda_cycle 10; lambda_identity 2.5 decays to 0"],
        ["Stability", "50-image pool; DiffAugment; EMA 0.9999; weight averaging; per-direction selection", "50-image pool; horizontal flip only (no DiffAugment); AMP; grad clip 10; EMA 0.999"],
        ["Schedule", "40-epoch base plus documented continuations to epoch 126; selected EMA 123", "150 epochs; G lr 2e-4, D lr 1e-4; linear decay from 75; best epoch 130"],
        ["Hardware", "RTX 4090, fp32", "RTX 4090, AMP"],
    ]
    story.append(styled_table(t3_design, [1.15 * inch, 2.75 * inch, 2.75 * inch], font_size=6.9))
    story.append(P("3.2 Official evaluation and Kaggle result", "H2"))
    official_rows = [
        ["Member", "A2B FID / MiFID", "B2A FID / MiFID", "Submission FID", "Submission MiFID", "Score", "Leaderboard evidence"],
        ["Chaitanya", f"{c3_official['FID_A2B']:.3f} / {c3_official['MiFID_A2B']:.4f}", f"{c3_official['FID_B2A']:.3f} / {c3_official['MiFID_B2A']:.4f}", f"{c3_official['FID']:.4f}", f"{c3_official['MiFID']:.4f}", "-48.2029", "Team submission; public -48.2029; team rank 15th"],
        ["Aswin", "109.561 / 0.4220", "100.668 / 0.4054", f"{aswin_fid:.4f}", f"{aswin_mifid:.4f}", f"{aswin_score:.4f}", "Submitted under the team 2026-10-06; public -52.7640"],
    ]
    story.append(styled_table(official_rows, [0.72 * inch, 0.92 * inch, 0.92 * inch, 0.77 * inch, 0.77 * inch, 0.68 * inch, 1.8 * inch], font_size=6.5))
    story.append(Spacer(1, 6))
    story.append(
        callout(
            "Team selection: Chaitanya final model",
            "Its official FID is 9.108 points lower and its MiFID is 0.0139 lower than Aswin V2, improving the competition score from "
            f"{aswin_score:.4f} to -48.2029. The documented progression from -53.83 to -48.20 came from DiffAugment, longer training, EMA, "
            "checkpoint averaging, weaker cycle/identity constraints, and direction-specific fine-tuning. The last 0.02-point gain is within snapshot "
            "noise and should not be overinterpreted.",
            "green",
        )
    )
    story.append(P("3.3 Full local metrics, both directions", "H2"))
    local_rows = [["Member / direction", "FID", "KID mean +/- std", "Precision / recall", "Density / coverage", "Cycle L1", "LPIPS", "Content cosine"]]
    for direction, label in [("A2B (Monet->photo)", "C A2B"), ("B2A (photo->Monet)", "C B2A")]:
        r = c3_by_dir[direction]
        local_rows.append([label, num(r["fid"]), f"{num(r['kid_mean'])} +/- {num(r['kid_std'])}", f"{num(r['gen_precision'], 3)} / {num(r['gen_recall'], 3)}", f"{num(r['density'], 3)} / {num(r['coverage'], 3)}", num(r["cycle_l1"]), num(r["lpips"]), num(r["content_cosine_sim"])])
    for direction, label in [("A2B_monet_to_photo", "A A2B"), ("B2A_photo_to_monet", "A B2A")]:
        r = a3_by_dir[direction]
        local_rows.append([label, num(r["fid"]), f"{num(r['kid_mean'])} +/- {num(r['kid_std'])}", f"{num(r['generative_precision'], 3)} / {num(r['generative_recall'], 3)}", "not computed", num(r["cycle_reconstruction_l1"]), num(r["lpips"]), num(r["content_cosine"])])
    story.append(styled_table(local_rows, [0.82 * inch, 0.62 * inch, 1.08 * inch, 1.0 * inch, 0.9 * inch, 0.68 * inch, 0.63 * inch, 0.83 * inch], font_size=6.4))
    story.append(
        P(
            "Local torchmetrics values and official notebook values differ because the feature implementations and sample protocols differ. "
            "Only the official notebook values are used for Kaggle selection; the local table is diagnostic. Chaitanya's final B2A model has "
            "higher recall and lower FID, while A2B has higher precision but lower recall. Aswin's B2A shows the same trade-off: recall 0.583 "
            "with precision 0.425."
        )
    )
    story.append(P("3.4 Stability, losses, and efficiency", "H2"))
    c_a2b = c3_by_dir["A2B (Monet->photo)"]
    c_b2a = c3_by_dir["B2A (photo->Monet)"]
    a_first = a3[0]
    stability = [
        ["Item", "Chaitanya final lineage", "Aswin V2"],
        ["Final G / D losses", f"G {num(c_a2b['final_g_loss'], 3)}; D_A {num(c_a2b['final_d_loss'], 3)}, D_B {num(c_b2a['final_d_loss'], 3)}", "Checkpoint G 3.214; D_A 0.129; D_B 0.179"],
        ["Cycle / identity loss", f"A2B {num(c_a2b['cycle_loss'], 3)}/{num(c_a2b['identity_loss'], 3)}; B2A {num(c_b2a['cycle_loss'], 3)}/{num(c_b2a['identity_loss'], 3)}", "Checkpoint cycle 0.229; identity 0.463"],
        ["Gradient / non-finite", f"{c_a2b['max_grad_norm']}; NaNs {c_a2b['nan_count']}", f"G mean {num(a_first['generator_grad_norm_mean'], 2)}; D mean {num(a_first['discriminator_grad_norm_mean'], 2)}; NaNs 0; skipped non-finite steps 70"],
        ["Parameters", "11,378,179 per generator", "28,299,912 total; 11,383,427 per generator"],
        ["Training time", f"{float(c_a2b['train_time_s'])/3600:.1f} h cumulative lineage", f"{float(a_first['training_time_seconds'])/3600:.2f} h"],
        ["Throughput", f"A2B {num(c_a2b['images_per_sec'], 1)}; B2A {num(c_b2a['images_per_sec'], 1)} images/s (generation)", f"{num(a_first['training_images_per_second'], 2)} train images/s"],
        ["Peak GPU memory", f"{float(c_a2b['peak_memory_mb'])/1024:.1f} GB", f"{float(a_first['peak_memory_mb'])/1024:.1f} GB"],
    ]
    story.append(styled_table(stability, [1.25 * inch, 2.7 * inch, 2.7 * inch], font_size=6.8))
    story.append(
        two_figures(
            "task3_gan/member_chaitanya/outputs/full_run18/loss_curves.png",
            "Chaitanya run 18: generator, discriminator, cycle, and identity losses.",
            "task3_gan/member_aswin/outputs/plots/aswin_cyclegan_v2_training_curves.png",
            "Aswin V2: generator, discriminator, and cycle losses over 150 epochs; no NaN losses, 70 skipped non-finite gradient steps.",
        )
    )
    story.extend(figure("task3_gan/member_aswin/outputs/plots/aswin_cyclegan_v2_validation_fid.png", "Aswin V2 validation B2A FID every 5 epochs; minimum 189.55 at epoch 130 selected best_model.pt (validation protocol, not the official one).", max_h=2.6 * inch))
    story.append(P("3.5 Qualitative evidence and cycle consistency", "H2"))
    story.extend(figure("task3_gan/member_chaitanya/outputs/full_run18/grids/epoch_123.png", "Chaitanya selected checkpoint. Rows 1-3 show real Monet, Monet->photo, and cycle reconstruction; rows 4-6 show real photo, photo->Monet, and cycle reconstruction.", max_h=7.1 * inch))
    story.append(PageBreak())
    story.extend(figure("task3_gan/member_aswin/outputs/plots/aswin_cyclegan_v2_qualitative_grid.png", "Aswin V2 fixed qualitative grid. The row order is printed in the image: real A, fake B, cycle A, real B, fake A, cycle B.", max_h=7.1 * inch))
    story.append(P("3.6 Failure analysis", "H2"))
    failures3 = [
        ("task3_gan/member_chaitanya/outputs/failure_cases/B2A_high_cycle_error/1_d0a1eed9dd.png", "Photo->Monet high cycle error: the faded pink input is recolored green/blue and reconstructs yellow-brown. Structure remains, but color information is discarded."),
        ("task3_gan/member_chaitanya/outputs/failure_cases/B2A_low_content/1_3a7a0992dd.png", "Photo->Monet lowest content similarity: an out-of-distribution night scene becomes daylight; point lights blob and skyline detail dissolves."),
        ("task3_gan/member_chaitanya/outputs/failure_cases/A2B_high_cycle_error/1_4f7e01f097.png", "Monet->photo high cycle error: dense brushwork becomes streaky pseudo-photographic texture; signature and fine detail cannot be reconstructed."),
    ]
    for rel, caption in failures3:
        story.extend(figure(rel, caption, max_h=2.25 * inch))
    story.append(
        P(
            "Each strip is input | translation | reconstruction and was selected by a fixed metric-based ranking, not by visual cherry-picking. "
            "The cases show why low cycle error is necessary but not sufficient: the model can preserve reconstructable structure while hiding or "
            "discarding color and semantic details. Aswin's failure_analysis.md records eight training and evaluation issues with their fixes (AMP non-finite gradients skipped with zero NaN losses, best-checkpoint selection at epoch 130, provenance checks on the prediction folders). Visually, V2 shows a cool blue-green colour bias, grainy dark scenes, and conservative translation, although raters scored V2 higher on artifacts than the submitted model (Section 3.7)."
        )
    )
    story.append(P("3.7 Blinded human audit (30 fixed inputs x 2 models, 2 raters)", "H2"))
    audit_root = ROOT / "task3_gan/member_chaitanya/outputs/human_audit"
    with (audit_root / "audit_results.json").open(encoding="utf-8") as stream:
        audit_all = json.load(stream)
    story.append(
        P(
            "30 inputs were fixed with seed 266 (20 photo-to-Monet, 10 Monet-to-photo; human_audit/inputs.txt). Each input was translated by "
            "both final models (Chaitanya run 18 EMA 123; Aswin V2 epoch 130), giving 60 items that were shuffled and shown under anonymous IDs "
            "with no model names. Both members rated every item independently on a 1-5 scale for style, content preservation, and artifacts "
            "(5 = no visible artifacts). The answer key (key.csv) was opened only after both raters had submitted."
        )
    )
    audit_rows = [["Model / criterion", "Mean", "Rater 1 / Rater 2", "B2A / A2B mean", "Cohen's kappa (unweighted / quadratic)", "Exact / within-1 agreement"]]
    for key, label in [("chaitanya_submitted", "Chaitanya"), ("aswin_v2", "Aswin V2")]:
        for name in ["style", "content", "artifacts"]:
            a = audit_all[key][name]
            audit_rows.append([
                f"{label}: {name}", num(a["mean"], 2), f"{num(a['rater_1_mean'], 2)} / {num(a['rater_2_mean'], 2)}",
                f"{num(a['mean_B2A'], 2)} / {num(a['mean_A2B'], 2)}", f"{num(a['kappa'], 3)} / {num(a['kappa_quadratic'], 3)}",
                f"{a['pct_exact_agreement']:.1f}% / {a['pct_within_1']:.1f}%",
            ])
    for name in ["style", "content", "artifacts"]:
        a = audit_all["pooled_all_60_items"][name]
        audit_rows.append([f"All 60 items: {name}", "-", "-", "-", f"{num(a['kappa'], 3)} / {num(a['kappa_quadratic'], 3)}", f"{a['pct_exact_agreement']:.1f}% / {a['pct_within_1']:.1f}%"])
    story.append(styled_table(audit_rows, [1.25 * inch, 0.5 * inch, 1.0 * inch, 0.95 * inch, 1.6 * inch, 1.4 * inch], font_size=6.7))
    story.append(Spacer(1, 5))
    paired = [["Paired by input", "Mean diff (Chaitanya - Aswin)", "Chaitanya higher / Aswin higher / tie", "Wilcoxon p"]]
    for name in ["style", "content", "artifacts"]:
        a = audit_all["paired_by_input"][name]
        paired.append([name.capitalize(), num(a["mean_diff_chaitanya_minus_aswin"], 2), f"{a['inputs_chaitanya_higher']} / {a['inputs_aswin_higher']} / {a['ties']}", num(a["wilcoxon_p"], 3)])
    story.append(styled_table(paired, [1.25 * inch, 1.6 * inch, 2.1 * inch, 1.0 * inch], font_size=6.9))
    story.append(
        P(
            "Both models score above 4.5 on style and content, and the paired differences there are not significant (p 0.26 and 0.74). The "
            "one significant difference is artifacts: Aswin V2 is rated cleaner on 19 of 30 inputs (4.68 vs 4.32, p = 0.002). This is the "
            "opposite of the FID ranking, which favours Chaitanya's model by 9 points: the looser cycle constraint that moves outputs closer "
            "to the Monet distribution also introduces visible artifacts. Inter-rater agreement is low (pooled quadratic kappa 0.25 style, "
            "0.02 content, -0.20 artifacts) even though 83-98% of ratings fall within one point; with most scores at 4-5 there is little "
            "variance for kappa to credit. A shared rubric with anchor images is the clearest fix. A first round that rated only Chaitanya's "
            "model was superseded because the rater could identify their own model (archived in human_audit/round1_own_model_only/)."
        )
    )
    audit_dir = audit_root / "images"
    examples = [("ca62c55ebe (photo->Monet)", "S05", "S01"), ("d1d9748a64 (Monet->photo)", "S08", "S09")]
    img_row, cap_row = [], []
    for label, c_item, a_item in examples:
        for path, cap in [(f"{a_item}_in.jpg", f"{label} input"), (f"{c_item}_out.jpg", "Chaitanya output"), (f"{a_item}_out.jpg", "Aswin V2 output")]:
            img_row.append(scaled_image(audit_dir / path, 1.05 * inch, 1.05 * inch))
            cap_row.append(P(cap, "Caption"))
    audit_imgs = Table([img_row, cap_row], colWidths=[1.12 * inch] * 6)
    audit_imgs.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(audit_imgs)
    story.append(P("task3_gan/member_chaitanya/outputs/human_audit/images/ (items S01, S05, S08, S09; model mapping in key.csv; ratings in rater_1.csv, rater_2.csv)", "CodePath"))
    story.append(P("3.8 Joint analysis and shortcomings", "H2"))
    story.append(
        P(
            "The best model improved the official distribution score by relaxing content constraints and specializing each direction. This is "
            "effective for FID but introduces a real trade-off: lower lambda_cycle and zero identity weight permit stronger style movement while "
            "weakening color and content preservation. EMA and weight averaging improve snapshot stability without changing inference architecture. "
            "The failed self-attention/spectral-normalization trials and non-transferable fine-tunes show that adding capacity or stabilization can "
            "unbalance the adversarial game. Aswin V2 is much cheaper and completes 150 epochs in 2.66 hours with about 4 GB peak memory, but its "
            "stronger cycle constraint (lambda_cycle 10) likely keeps outputs closer to inputs, and its discriminators train on un-augmented images: "
            "with only 300 paintings the Monet discriminator can memorise the training set, which DiffAugment counters (DiffAugment was "
            "Chaitanya's largest single gain, -53.87 to -51.47). Both factors plausibly explain V2's weaker official distribution match, but "
            "they were not ablated separately. Future work should "
            "pre-register a held-out model-selection subset, ablate one factor at a time, rate with a shared anchored rubric, and test a discriminator-scale "
            "ablation with identical schedules."
        )
    )
    story.append(
        callout(
            "Open item: private leaderboard",
            "Both submissions and the blinded two-model human audit are recorded. The private leaderboard scores become available only "
            "after the competition closes; add them here then.",
            "amber",
        )
    )
    story.append(P("3.9 Evidence map", "H2"))
    add_evidence_table(
        story,
        [
            ("Official team evaluation", "task3_gan/member_chaitanya/outputs/official_eval.json and submission.csv"),
            ("Chaitanya metrics", "task3_gan/member_chaitanya/full_metrics_report.csv"),
            ("Chaitanya design lineage", "task3_gan/member_chaitanya/DESIGN_LOG.md"),
            ("Chaitanya checkpoint", "task3_gan/member_chaitanya/checkpoints/full_run18/epoch_123_ema/ (local, not in Git; source in outputs/pred_source.txt)"),
            ("Human audit", "task3_gan/member_chaitanya/outputs/human_audit/ (key.csv, rater_1.csv, rater_2.csv, audit_results.json); superseded round 1 in round1_own_model_only/"),
            ("Aswin official evaluation", "task3_gan/member_aswin/src/Part3_Evaluation_Script_evaluated.ipynb and submission.csv"),
            ("Aswin metrics", "task3_gan/member_aswin/full_metrics_report.csv"),
            ("Aswin checkpoint", "task3_gan/member_aswin/checkpoints/aswin_cyclegan_v2/best_model.pt (epoch 130)"),
        ],
    )
    story.append(PageBreak())

    # Reproducibility and conclusion
    story.append(P("4. Reproducibility and professional practice", "H1"))
    repro = [
        ["Requirement", "Evidence / status"],
        ["Named member folders", "Present for both members under all three task directories."],
        ["Executed notebooks", "Preserved in each member's src folder; Task 3 official evaluation notebooks include outputs."],
        ["Metrics and analysis", "CSV metrics, results.md, failure_analysis.md, plots, predictions, and checkpoint identifiers are linked above."],
        ["Raw logs and manifests", "Committed under reproducibility/raw_logs and reproducibility/manifests where available; Aswin run logs are also retained in member output folders."],
        ["Smoke-test entry points", "Top-level README documents smoke-test commands and artifact locations."],
        ["Personal paths / secrets", "Final report uses repository-relative evidence paths; no credential is embedded."],
        ["Known incomplete evidence", "Private leaderboard results (after competition close)."],
    ]
    story.append(styled_table(repro, [1.7 * inch, 4.95 * inch], font_size=7.2))
    story.append(P("4.1 Threats to validity", "H2"))
    story.append(bullet("Task 1 split, cleaning, exposure, vocabulary, and distinct-n definitions differ; model-size conclusions are confounded."))
    story.append(bullet("Task 2 preprocessing and seed counts differ across members; architecture is not the only changed variable."))
    story.append(bullet("Task 3 repeatedly selects checkpoints on the same fixed official images; reported gains can be optimistically biased and small differences may be noise."))
    story.append(bullet("Hardware and precision differ, so wall-clock comparisons measure full systems rather than architecture alone."))
    story.append(P("4.2 Final synthesis", "H2"))
    story.append(
        P(
            "Across all tasks, the strongest results come from matching inductive bias and evaluation to the data. More context and more training "
            "exposure improve the character GPT; local n-gram structure dominates sentiment at 20K examples; and EMA plus direction-specific "
            "specialization improves CycleGAN distribution matching. The compact alternatives remain important: Aswin's GPT and CycleGAN deliver "
            "large efficiency gains, and Aswin's selected TextCNN is statistically indistinguishable from the highest-scoring sentiment model under the paired "
            "team comparison. The final team recommendation is therefore score-led but evidence-aware: report the selected models, retain both "
            "members' independent artifacts, and present efficiency, confounds, and failure behavior alongside headline metrics."
        )
    )
    # Appendix A: complete error audit, landscape.
    switch_landscape(story)
    story.append(P("Appendix A - Complete Task 2 error-review index", "H1"))
    story.append(
        P(
            "These compact tables include all 20 reviewed errors per member. Excerpts are shortened only for layout; complete unedited review text, "
            "model-visible tokens, explanations, and proposed fixes remain in the cited CSV files."
        )
    )
    c_errors = read_csv("task2_sentiment/member_chaitanya/outputs/error_review/baseline_candidates.csv")
    c_err_table = [["#", "Category", "True/pred", "P(pos)", "Slices", "Actual review excerpt"]]
    for row in c_errors:
        c_err_table.append([row["n"], row["category"], f"{row['label']}/{row['pred']}", num(row["prob_pos"], 4), compact(row["slices"], 75), compact(row["text"], 190)])
    story.append(P("<b>A.1 Chaitanya fastText-bigram baseline</b>", "Small"))
    story.append(styled_table(c_err_table, [0.35 * inch, 1.05 * inch, 0.6 * inch, 0.55 * inch, 1.55 * inch, 6.0 * inch], font_size=5.45, landscape_mode=True, padding=2.2))
    a_errors = read_csv("task2_sentiment/member_aswin/outputs/error_analysis/error_analysis.csv")
    a_err_table = [["#", "Category", "True/pred", "P(pos)", "Error type", "Actual review excerpt", "Proposed fix"]]
    for row in a_errors:
        a_err_table.append([row["idx"], row["category"], f"{row['true_label']}/{row['pred_label']}", num(row["prob_positive"], 4), compact(row["error_type"], 60), compact(row["full_text"], 155), compact(row["proposed_fix"], 120)])
    story.append(P("<b>A.2 Aswin best-model audit</b>", "Small"))
    story.append(styled_table(a_err_table, [0.3 * inch, 1.15 * inch, 0.55 * inch, 0.5 * inch, 1.25 * inch, 4.05 * inch, 2.35 * inch], font_size=5.2, landscape_mode=True, padding=2.2))
    switch_portrait(story)

    # References
    story.append(P("References", "H1"))
    refs = [
        "Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 30. arXiv:1706.03762.",
        "Eldan, R., and Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English? arXiv:2305.07759.",
        "Zhu, J.-Y., Park, T., Isola, P., and Efros, A. A. (2017). Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks. ICCV.",
        "Yang, Z. et al. (2016). Hierarchical Attention Networks for Document Classification. NAACL-HLT.",
        "Joulin, A. et al. (2017). Bag of Tricks for Efficient Text Classification. EACL.",
        "Zhao, S. et al. (2020). Differentiable Augmentation for Data-Efficient GAN Training. NeurIPS.",
        "Zhang, X., Zhao, J., and LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. NeurIPS.",
        "Holtzman, A. et al. (2020). The Curious Case of Neural Text Degeneration. ICLR.",
    ]
    for ref in refs:
        story.append(bullet(ref))
    story.append(Spacer(1, 10))
    story.append(P("Repository", "H2"))
    story.append(P('<link href="https://github.com/ChaitanyaUppalapati/Lab_Pair_18" color="#2166AC">https://github.com/ChaitanyaUppalapati/Lab_Pair_18</link>'))
    story.append(P("All numerical claims in this report are backed by the repository paths shown in the evidence tables.", "Small"))
    return story


def build_pdf() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    portrait_margin = 0.62 * inch
    landscape_margin = 0.52 * inch
    portrait_frame = Frame(
        portrait_margin,
        0.52 * inch,
        A4[0] - 2 * portrait_margin,
        A4[1] - 1.05 * inch,
        id="portrait-frame",
        topPadding=8,
        bottomPadding=8,
        leftPadding=0,
        rightPadding=0,
    )
    land_size = landscape(A4)
    landscape_frame = Frame(
        landscape_margin,
        0.52 * inch,
        land_size[0] - 2 * landscape_margin,
        land_size[1] - 1.05 * inch,
        id="landscape-frame",
        topPadding=8,
        bottomPadding=8,
        leftPadding=0,
        rightPadding=0,
    )
    doc = ReportDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        title="DATA266 Lab 1 - Team 18 Joint Technical Report",
        author="Aswin John and Chaitanya Uppalapati",
        subject="Joint comparison of character GPT, Yelp sentiment, and CycleGAN experiments",
        creator="Repository evidence report builder",
        leftMargin=portrait_margin,
        rightMargin=portrait_margin,
        topMargin=0.55 * inch,
        bottomMargin=0.52 * inch,
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="portrait", pagesize=A4, frames=[portrait_frame], onPage=page_header_footer),
            PageTemplate(id="landscape", pagesize=land_size, frames=[landscape_frame], onPage=page_header_footer),
        ]
    )
    doc.multiBuild(build_story())


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT)
