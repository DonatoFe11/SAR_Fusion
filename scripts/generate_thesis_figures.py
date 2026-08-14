#!/usr/bin/env python3
"""Generate the five portrait-native figures used by the Search and Rescue thesis.

The script intentionally depends only on pycairo, which is already available in
the project environment. Text remains vector text in the four PDF outputs; the
P5 diagnostic is a high-resolution raster assembled from saved experiment data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cairo


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes" / "Search_and_Rescue" / "images"
PREVIEW_OUT = Path("/tmp/thesis_figure_previews")

FONT = "DejaVu Sans"
NAVY = (0.105, 0.239, 0.373)
BLUE = (0.137, 0.533, 0.722)
LIGHT_BLUE = (0.879, 0.946, 0.976)
ORANGE = (0.902, 0.424, 0.102)
LIGHT_ORANGE = (0.992, 0.916, 0.843)
TEAL = (0.000, 0.455, 0.443)
LIGHT_TEAL = (0.850, 0.953, 0.941)
PURPLE = (0.396, 0.302, 0.635)
LIGHT_PURPLE = (0.925, 0.902, 0.957)
GREEN = (0.180, 0.545, 0.341)
LIGHT_GREEN = (0.882, 0.949, 0.902)
RED = (0.745, 0.153, 0.180)
LIGHT_RED = (0.984, 0.890, 0.890)
GOLD = (0.824, 0.643, 0.125)
LIGHT_GOLD = (0.980, 0.949, 0.824)
INK = (0.105, 0.125, 0.145)
MID = (0.390, 0.425, 0.455)
GRID = (0.825, 0.845, 0.860)
WHITE = (1.0, 1.0, 1.0)


def set_rgb(ctx: cairo.Context, colour):
    ctx.set_source_rgb(*colour)


def text_width(ctx: cairo.Context, value: str) -> float:
    return ctx.text_extents(value).x_advance


def wrap_text(ctx: cairo.Context, value: str, max_width: float) -> list[str]:
    lines: list[str] = []
    for paragraph in value.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if text_width(ctx, candidate) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_text(
    ctx: cairo.Context,
    value: str,
    x: float,
    y: float,
    *,
    size: float = 12,
    colour=INK,
    bold: bool = False,
    align: str = "left",
    max_width: float | None = None,
    line_height: float | None = None,
):
    ctx.select_font_face(
        FONT,
        cairo.FONT_SLANT_NORMAL,
        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL,
    )
    ctx.set_font_size(size)
    set_rgb(ctx, colour)
    lines = wrap_text(ctx, value, max_width) if max_width else value.split("\n")
    spacing = line_height or size * 1.28
    for index, line in enumerate(lines):
        width = text_width(ctx, line)
        tx = x
        if align == "center":
            tx = x - width / 2
        elif align == "right":
            tx = x - width
        ctx.move_to(tx, y + index * spacing)
        ctx.show_text(line)
    return len(lines) * spacing


def rounded_rect(ctx, x, y, width, height, radius=10):
    r = min(radius, width / 2, height / 2)
    ctx.new_sub_path()
    ctx.arc(x + width - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + width - r, y + height - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + height - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def box(
    ctx,
    x,
    y,
    width,
    height,
    label,
    *,
    fill=WHITE,
    stroke=INK,
    size=11,
    bold=False,
    radius=9,
    line_width=1.4,
):
    rounded_rect(ctx, x, y, width, height, radius)
    set_rgb(ctx, fill)
    ctx.fill_preserve()
    set_rgb(ctx, stroke)
    ctx.set_line_width(line_width)
    ctx.stroke()
    ctx.select_font_face(
        FONT,
        cairo.FONT_SLANT_NORMAL,
        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL,
    )
    ctx.set_font_size(size)
    lines = wrap_text(ctx, label, width - 16)
    spacing = size * 1.22
    first_y = y + height / 2 - (len(lines) - 1) * spacing / 2 + size * 0.34
    set_rgb(ctx, INK)
    for index, line in enumerate(lines):
        draw_text(
            ctx,
            line,
            x + width / 2,
            first_y + index * spacing,
            size=size,
            colour=INK,
            bold=bold,
            align="center",
        )


def arrow(ctx, x1, y1, x2, y2, *, colour=INK, width=1.6, dashed=False, head=7):
    set_rgb(ctx, colour)
    ctx.set_line_width(width)
    ctx.set_dash([6, 4] if dashed else [])
    ctx.move_to(x1, y1)
    ctx.line_to(x2, y2)
    ctx.stroke()
    ctx.set_dash([])
    angle = math.atan2(y2 - y1, x2 - x1)
    ctx.move_to(x2, y2)
    ctx.line_to(
        x2 - head * math.cos(angle - math.pi / 6),
        y2 - head * math.sin(angle - math.pi / 6),
    )
    ctx.line_to(
        x2 - head * math.cos(angle + math.pi / 6),
        y2 - head * math.sin(angle + math.pi / 6),
    )
    ctx.close_path()
    ctx.fill()


def line(ctx, x1, y1, x2, y2, *, colour=INK, width=1.4, dashed=False):
    set_rgb(ctx, colour)
    ctx.set_line_width(width)
    ctx.set_dash([6, 4] if dashed else [])
    ctx.move_to(x1, y1)
    ctx.line_to(x2, y2)
    ctx.stroke()
    ctx.set_dash([])


def circle(ctx, x, y, radius, *, fill=WHITE, stroke=INK, width=1.3):
    ctx.new_sub_path()
    ctx.arc(x, y, radius, 0, 2 * math.pi)
    set_rgb(ctx, fill)
    ctx.fill_preserve()
    set_rgb(ctx, stroke)
    ctx.set_line_width(width)
    ctx.stroke()


def marker(ctx, x, y, radius, shape, *, fill=WHITE, stroke=INK, width=1.3):
    """Draw a seed marker that remains identifiable without colour."""
    if shape == "circle":
        circle(ctx, x, y, radius, fill=fill, stroke=stroke, width=width)
        return

    if shape == "square":
        points = [
            (x - radius, y - radius),
            (x + radius, y - radius),
            (x + radius, y + radius),
            (x - radius, y + radius),
        ]
    elif shape == "diamond":
        points = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
    else:
        sides = 3 if shape == "triangle" else 5
        points = [
            (
                x + radius * math.cos(-math.pi / 2 + 2 * math.pi * index / sides),
                y + radius * math.sin(-math.pi / 2 + 2 * math.pi * index / sides),
            )
            for index in range(sides)
        ]
    ctx.new_path()
    ctx.move_to(*points[0])
    for point in points[1:]:
        ctx.line_to(*point)
    ctx.close_path()
    set_rgb(ctx, fill)
    ctx.fill_preserve()
    set_rgb(ctx, stroke)
    ctx.set_line_width(width)
    ctx.stroke()


def pdf_canvas(filename: str, width: float, height: float):
    OUT.mkdir(parents=True, exist_ok=True)
    recording = cairo.RecordingSurface(
        cairo.CONTENT_COLOR_ALPHA,
        cairo.Rectangle(0, 0, width, height),
    )
    ctx = cairo.Context(recording)
    set_rgb(ctx, WHITE)
    ctx.paint()
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    class FigureSurface:
        def finish(self):
            pdf_path = OUT / filename
            pdf = cairo.PDFSurface(str(pdf_path), width, height)
            pdf_ctx = cairo.Context(pdf)
            pdf_ctx.set_source_surface(recording, 0, 0)
            pdf_ctx.paint()
            pdf.finish()

            # Raster previews are kept outside the repository for visual QA.
            PREVIEW_OUT.mkdir(parents=True, exist_ok=True)
            preview = cairo.ImageSurface(
                cairo.FORMAT_ARGB32,
                int(width * 2),
                int(height * 2),
            )
            preview_ctx = cairo.Context(preview)
            preview_ctx.scale(2, 2)
            preview_ctx.set_source_surface(recording, 0, 0)
            preview_ctx.paint()
            preview.write_to_png(str(PREVIEW_OUT / f"{Path(filename).stem}.png"))

            recording.finish()

    return FigureSurface(), ctx


def title(ctx, value, width, subtitle=None, *, subtitle_size=9.5):
    draw_text(ctx, value, width / 2, 30, size=17, bold=True, align="center", colour=NAVY)
    if subtitle:
        draw_text(ctx, subtitle, width / 2, 49, size=subtitle_size, align="center", colour=MID)


def generate_timeline():
    """Render the study chronology as a portrait, top-to-bottom flow."""
    width, height = 680, 850
    surface, ctx = pdf_canvas("experimental_timeline.pdf", width, height)

    phases = [
        (
            LIGHT_BLUE,
            BLUE,
            "1  Preceding coursework",
            "Inherited starting point",
            "DETR + Modal Dropout\nRT-DETR Additive; tiling and CMX\nFAM on RT-DETR\nDeformable DETR + FAM",
        ),
        (
            LIGHT_ORANGE,
            ORANGE,
            "2  Post-coursework exploration",
            "Previous protocol",
            "Lazy / Eager / Frozen FAM\nSpatial Dropout and SSJ\nDeformable DETR + FAM + SSJ\nComplete DINO baseline",
        ),
        (
            LIGHT_PURPLE,
            PURPLE,
            "3  Reproducibility audit",
            "Protocol redesign",
            "CUDA variability and transferred person head\nValidation and checkpoint audit\nTraining-only Modal Dropout\nIsolated processes and paired seeds",
        ),
        (
            LIGHT_TEAL,
            TEAL,
            "4  Locked-protocol evaluation",
            "Principal thesis evidence",
            "RT-DETR: six configurations × five seeds\nDiagnostics, level ablation, bounded offsets\nError analysis, Carnation, compute\nYOLOv10: Additive versus FAM × five seeds",
        ),
    ]

    card_x, card_w, card_h = 92, 545, 165
    card_y = [28, 225, 422, 660]
    line(ctx, 53, 54, 53, 807, colour=NAVY, width=2.5)
    arrow(ctx, 53, 807, 53, 830, colour=NAVY, width=2.5, head=10)

    for index, ((fill, stroke, heading, tag, body), y) in enumerate(zip(phases, card_y)):
        centre_y = y + card_h / 2
        circle(ctx, 53, centre_y, 9, fill=stroke, stroke=WHITE, width=2)
        arrow(ctx, 62, centre_y, card_x, centre_y, colour=stroke, width=1.8, head=8)
        box(ctx, card_x, y, card_w, card_h, "", fill=fill, stroke=stroke, radius=14, line_width=2)
        draw_text(ctx, heading, card_x + 22, y + 31, size=18, bold=True, colour=stroke)
        box(
            ctx,
            card_x + card_w - 190,
            y + 14,
            170,
            32,
            tag,
            fill=WHITE,
            stroke=stroke,
            size=13,
            bold=True,
            radius=16,
        )
        line(ctx, card_x + 22, y + 55, card_x + card_w - 22, y + 55, colour=stroke, width=1.1)
        draw_text(ctx, body, card_x + 26, y + 82, size=15, line_height=22, max_width=card_w - 52)

        if index == 2:
            boundary_y = y + card_h + 28
            line(ctx, 32, boundary_y, 648, boundary_y, colour=RED, width=2.2, dashed=True)
            box(
                ctx,
                232,
                boundary_y - 18,
                216,
                36,
                "LOCKED PROTOCOL",
                fill=LIGHT_RED,
                stroke=RED,
                size=14,
                bold=True,
                radius=18,
            )

    surface.finish()


def draw_fam_inset(ctx, x, y, width, height, *, compact=False):
    box(ctx, x, y, width, height, "", fill=(0.985, 0.988, 0.991), stroke=PURPLE, radius=10, line_width=1.5)
    draw_text(ctx, "FAM at level l", x + width / 2, y + 22, size=15, bold=True, align="center", colour=PURPLE)
    if compact:
        box(ctx, x + 10, y + 40, 72, 31, "RGB(l)", fill=LIGHT_BLUE, stroke=BLUE, size=14, bold=True)
        box(ctx, x + 10, y + 91, 72, 31, "IR(l)", fill=LIGHT_ORANGE, stroke=ORANGE, size=14, bold=True)
        box(ctx, x + 100, y + 41, 86, 43, "Concat →\nConv 3×3", fill=LIGHT_PURPLE, stroke=PURPLE, size=13, bold=True)
        box(ctx, x + 205, y + 41, 90, 43, "Δ: 18\nσ(m): 9", fill=LIGHT_GOLD, stroke=GOLD, size=14)
        box(ctx, x + 205, y + 94, 90, 34, "DCNv2(IR)", fill=LIGHT_ORANGE, stroke=ORANGE, size=14, bold=True)
        circle(ctx, x + 330, y + 110, 12, fill=LIGHT_TEAL, stroke=TEAL)
        draw_text(ctx, "+", x + 330, y + 115, size=16, bold=True, align="center", colour=TEAL)
        draw_text(ctx, "F(l)", x + 370, y + 115, size=14, bold=True, colour=TEAL)
        arrow(ctx, x + 82, y + 55, x + 100, y + 55, colour=BLUE)
        arrow(ctx, x + 82, y + 106, x + 100, y + 73, colour=ORANGE)
        arrow(ctx, x + 186, y + 62, x + 205, y + 62, colour=PURPLE)
        arrow(ctx, x + 250, y + 84, x + 250, y + 94, colour=GOLD)
        arrow(ctx, x + 82, y + 106, x + 205, y + 111, colour=ORANGE)
        arrow(ctx, x + 295, y + 111, x + 318, y + 110, colour=ORANGE)
        line(ctx, x + 88, y + 55, x + 88, y + 136, colour=BLUE, width=1.2)
        line(ctx, x + 88, y + 136, x + 330, y + 136, colour=BLUE, width=1.2)
        arrow(ctx, x + 330, y + 136, x + 330, y + 122, colour=BLUE)
        arrow(ctx, x + 342, y + 110, x + 365, y + 110, colour=TEAL)
    else:
        # Reserved for a larger variant if needed.
        pass


def generate_rtdetr_architecture():
    """Render RT-DETR as a portrait-native top-to-bottom pipeline."""
    width, height = 650, 920
    surface, ctx = pdf_canvas("rtdetr_fam_architecture.pdf", width, height)
    box(
        ctx,
        100,
        22,
        450,
        58,
        "Paired input: RGB (3 ch) + IR (1 ch)",
        fill=(0.96, 0.97, 0.98),
        stroke=NAVY,
        size=15,
        bold=True,
    )
    arrow(ctx, 325, 80, 325, 105, colour=NAVY)
    box(
        ctx,
        120,
        105,
        410,
        70,
        "Training-only Modal Dropout\nVIS 20% | IR 20% | VIS+IR 60%\nAbsent input channels are zero-padded",
        fill=(0.97, 0.97, 0.97),
        stroke=MID,
        size=14,
    )

    box(ctx, 30, 210, 270, 82, "RGB backbone\nResNet-50vd", fill=LIGHT_BLUE, stroke=BLUE, size=16, bold=True)
    box(ctx, 350, 210, 270, 82, "IR backbone\nResNet-50vd (1 ch)", fill=LIGHT_ORANGE, stroke=ORANGE, size=16, bold=True)
    line(ctx, 325, 175, 325, 190, colour=NAVY, width=1.8)
    arrow(ctx, 325, 190, 165, 210, colour=BLUE, width=1.8)
    arrow(ctx, 325, 190, 485, 210, colour=ORANGE, width=1.8)
    draw_text(ctx, "independent parameters", 325, 313, size=14, align="center", colour=MID)

    levels = [
        ("P3", "512 ch\n80 × 80 | s=8", 340),
        ("P4", "1024 ch\n40 × 40 | s=16", 425),
        ("P5", "2048 ch\n20 × 20 | s=32", 510),
    ]
    fused_centres = []
    for level, metadata, y in levels:
        box(ctx, 28, y, 125, 28, f"RGB {level}", fill=LIGHT_BLUE, stroke=BLUE, size=13, bold=True)
        box(ctx, 28, y + 38, 125, 28, f"IR {level}", fill=LIGHT_ORANGE, stroke=ORANGE, size=13, bold=True)
        box(ctx, 174, y + 10, 130, 46, metadata, fill=(0.965, 0.970, 0.976), stroke=GRID, size=13)
        box(ctx, 325, y + 3, 175, 60, f"FAM {level}\ntransform IR", fill=LIGHT_TEAL, stroke=TEAL, size=14, bold=True)
        circle(ctx, 535, y + 33, 14, fill=LIGHT_GOLD, stroke=GOLD)
        draw_text(ctx, "+", 535, y + 39, size=18, bold=True, align="center", colour=GOLD)
        box(ctx, 570, y + 9, 66, 48, f"fused\n{level}", fill=LIGHT_TEAL, stroke=TEAL, size=13, bold=True)
        # Route the two modalities around the metadata box.  The upper RGB
        # route also carries the residual branch to the additive fusion node.
        line(ctx, 153, y + 14, 162, y + 14, colour=BLUE)
        line(ctx, 162, y + 14, 162, y - 4, colour=BLUE)
        line(ctx, 162, y - 4, 520, y - 4, colour=BLUE)
        arrow(ctx, 310, y - 4, 325, y + 18, colour=BLUE, head=7)
        arrow(ctx, 520, y - 4, 528, y + 21, colour=BLUE, head=7)
        line(ctx, 153, y + 52, 162, y + 52, colour=ORANGE)
        line(ctx, 162, y + 52, 162, y + 70, colour=ORANGE)
        line(ctx, 162, y + 70, 310, y + 70, colour=ORANGE)
        arrow(ctx, 310, y + 70, 325, y + 48, colour=ORANGE, head=7)
        arrow(ctx, 500, y + 34, 521, y + 33, colour=TEAL, head=7)
        arrow(ctx, 549, y + 33, 570, y + 33, colour=TEAL, head=7)
        fused_centres.append(y + 33)

    # Fused levels converge into the standard RT-DETR encoder/decoder path.
    for centre_y in fused_centres:
        line(ctx, 636, centre_y, 642, centre_y, colour=TEAL, width=2)
    line(ctx, 642, fused_centres[0], 642, 620, colour=TEAL, width=2)
    line(ctx, 642, 620, 100, 620, colour=TEAL, width=2)
    arrow(ctx, 100, 620, 100, 655, colour=TEAL, width=2, head=8)

    box(ctx, 18, 655, 165, 68, "Efficient Hybrid\nEncoder", fill=LIGHT_GREEN, stroke=GREEN, size=14, bold=True)
    box(ctx, 214, 655, 120, 68, "Top-300\nqueries", fill=LIGHT_GOLD, stroke=GOLD, size=14, bold=True)
    box(ctx, 365, 655, 120, 68, "6-layer\ndecoder", fill=LIGHT_PURPLE, stroke=PURPLE, size=14, bold=True)
    box(ctx, 516, 655, 116, 68, "Person\ndetections", fill=LIGHT_TEAL, stroke=TEAL, size=14, bold=True)
    arrow(ctx, 183, 689, 214, 689, colour=GREEN)
    arrow(ctx, 334, 689, 365, 689, colour=GOLD)
    arrow(ctx, 485, 689, 516, 689, colour=PURPLE)

    draw_fam_inset(ctx, 120, 755, 410, 145, compact=True)
    surface.finish()


def generate_rtdetr_results():
    """Render the paired results as two vertically stacked portrait panels."""
    width, height = 650, 830
    surface, ctx = pdf_canvas("rtdetr_final_paired_map50.pdf", width, height)

    seeds = [40, 41, 42, 43, 44]
    colours = [BLUE, ORANGE, TEAL, PURPLE, RED]
    marker_shapes = ["circle", "square", "triangle", "diamond", "pentagon"]
    additive = [0.2566, 0.4029, 0.2958, 0.3476, 0.2372]
    fam = [0.3783, 0.4335, 0.3129, 0.3964, 0.3690]
    variants = {
        "IR Dropout": [0.4087, 0.3234, 0.4452, 0.3986, 0.3598],
        "SSJ": [0.3663, 0.3734, 0.3528, 0.4030, 0.3792],
        "Identity DCNv2": [0.4336, 0.2113, 0.2926, 0.2860, 0.4162],
        "Grid Sample": [0.4050, 0.3483, 0.4083, 0.3326, 0.3564],
    }

    # Panel A: paired slopegraph.
    panel_a_top, panel_a_bottom = 62, 350
    x_left, x_right = 225, 510
    y_min, y_max = 0.20, 0.46

    def map_y(value):
        return panel_a_bottom - (value - y_min) / (y_max - y_min) * (panel_a_bottom - panel_a_top)

    draw_text(ctx, "A  VIS+IR mAP@50: FAM versus Additive", 28, 31, size=16, bold=True, colour=NAVY)
    for tick in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        y = map_y(tick)
        line(ctx, 105, y, 600, y, colour=GRID, width=0.9)
        draw_text(ctx, f"{tick:.2f}", 94, y + 5, size=13, align="right", colour=MID)
    draw_text(ctx, "Additive", x_left, 385, size=15, bold=True, align="center")
    draw_text(ctx, "FAM", x_right, 385, size=15, bold=True, align="center")
    for seed, colour, shape, left_value, right_value in zip(seeds, colours, marker_shapes, additive, fam):
        y1, y2 = map_y(left_value), map_y(right_value)
        line(ctx, x_left, y1, x_right, y2, colour=colour, width=2.0)
        marker(ctx, x_left, y1, 6.5, shape, fill=WHITE, stroke=colour, width=2)
        marker(ctx, x_right, y2, 6.5, shape, fill=colour, stroke=colour, width=1)
        draw_text(ctx, str(seed), x_left - 13, y1 + 5, size=13, align="right", colour=colour)

    # Panel B: per-seed deltas relative to FAM.
    draw_text(ctx, "B  Delta mAP@50 relative to standard FAM", 28, 438, size=16, bold=True, colour=NAVY)
    x0, y0, plot_w, plot_h = 82, 468, 535, 245
    dmin, dmax = -0.24, 0.14

    def delta_y(value):
        return y0 + plot_h - (value - dmin) / (dmax - dmin) * plot_h

    zero_y = delta_y(0)
    for tick in [-0.20, -0.10, 0.00, 0.10]:
        yy = delta_y(tick)
        line(ctx, x0, yy, x0 + plot_w, yy, colour=INK if tick == 0 else GRID, width=1.4 if tick == 0 else 0.8)
        draw_text(ctx, f"{tick:+.2f}", x0 - 10, yy + 5, size=13, align="right", colour=MID)
    names = list(variants)
    display_names = ["IR Dropout", "SSJ", "Identity\nDCNv2", "Grid\nSample"]
    group_x = [145, 290, 435, 575]
    for gx, name in zip(group_x, display_names):
        draw_text(ctx, name, gx, 756, size=13, bold=True, align="center", max_width=120, line_height=16)
    offsets = [-28, -14, 0, 14, 28]
    for gx, name in zip(group_x, names):
        values = variants[name]
        deltas = [value - reference for value, reference in zip(values, fam)]
        mean_delta = sum(deltas) / len(deltas)
        for offset, colour, shape, delta in zip(offsets, colours, marker_shapes, deltas):
            marker(ctx, gx + offset, delta_y(delta), 5.5, shape, fill=colour, stroke=WHITE, width=1.2)
        line(ctx, gx - 36, delta_y(mean_delta), gx + 36, delta_y(mean_delta), colour=INK, width=3.0)
        draw_text(ctx, f"{mean_delta:+.4f}", gx, 735, size=13, align="center", colour=MID)

    # Legend.
    legend_x = 76
    for index, (seed, colour, shape) in enumerate(zip(seeds, colours, marker_shapes)):
        lx = legend_x + index * 105
        marker(ctx, lx, 805, 5.5, shape, fill=colour, stroke=colour)
        draw_text(ctx, f"seed {seed}", lx + 12, 810, size=13, colour=INK)
    surface.finish()


def draw_image_crop(ctx, source, sx, sy, sw, sh, dx, dy, dw, dh):
    """Draw one crop from a Cairo image surface into a bordered panel."""
    ctx.save()
    ctx.rectangle(dx, dy, dw, dh)
    ctx.clip()
    ctx.translate(dx, dy)
    ctx.scale(dw / sw, dh / sh)
    ctx.set_source_surface(source, -sx, -sy)
    ctx.get_source().set_filter(cairo.FILTER_BEST)
    ctx.paint()
    ctx.restore()
    rounded_rect(ctx, dx, dy, dw, dh, 8)
    set_rgb(ctx, GRID)
    ctx.set_line_width(2)
    ctx.stroke()


def generate_p5_collapse():
    """Repackage the prespecified P5 diagnostic as a portrait 2-column grid."""
    source_path = (
        ROOT
        / "out"
        / "rtdetr_fam_diagnostics"
        / "figures"
        / "fam"
        / "seed_41"
        / "mt_erie_p5"
        / "fam_sample35_level_2.png"
    )
    metrics_path = (
        ROOT
        / "out"
        / "rtdetr_fam_diagnostics"
        / "fam_seed41_mt_erie_sample35_p5.json"
    )
    if not source_path.is_file() or not metrics_path.is_file():
        missing = [str(path) for path in (source_path, metrics_path) if not path.is_file()]
        raise FileNotFoundError("Missing P5 diagnostic source: " + ", ".join(missing))

    source = cairo.ImageSurface.create_from_png(str(source_path))
    with metrics_path.open(encoding="utf-8") as input_file:
        row = json.load(input_file)["rows"][0]
    metrics = row["metrics"]
    offset = metrics["offset"]
    activation = metrics["activations"]["fam_ir"]

    width, height = 1600, 2050
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    set_rgb(ctx, WHITE)
    ctx.paint()
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    panel_titles = [
        "(a) RGB (shared PCA)",
        "(b) IR pre-FAM (shared PCA)",
        "(c) FAM(IR) (shared PCA)",
        "(d) Pre-FAM overlay",
        "(e) Post-FAM overlay",
    ]
    panel_positions = [
        (20, 0),
        (820, 0),
        (20, 680),
        (820, 680),
        (20, 1360),
    ]
    # The original six-panel diagnostic uses equal 600 px columns.  Cropping
    # below the legacy headings preserves the actual PCA maps while allowing
    # scientifically precise labels and an A4-readable arrangement.
    for index, ((panel_x, panel_y), panel_title) in enumerate(
        zip(panel_positions, panel_titles)
    ):
        draw_text(
            ctx,
            panel_title,
            panel_x + 380,
            panel_y + 43,
            size=36,
            bold=True,
            align="center",
            colour=NAVY,
        )
        draw_image_crop(
            ctx,
            source,
            index * 600 + 20,
            125,
            560,
            475,
            panel_x + 80,
            panel_y + 72,
            600,
            510,
        )

    panel_x, panel_y = 820, 1360
    draw_text(
        ctx,
        "(f) P5 collapse summary",
        panel_x + 380,
        panel_y + 43,
        size=36,
        bold=True,
        align="center",
        colour=NAVY,
    )
    box(
        ctx,
        panel_x + 35,
        panel_y + 72,
        710,
        615,
        "",
        fill=(0.978, 0.981, 0.985),
        stroke=PURPLE,
        radius=18,
        line_width=3,
    )
    draw_text(
        ctx,
        "Absolute offset components\n(P5 cells, log scale)",
        panel_x + 390,
        panel_y + 126,
        size=34,
        bold=True,
        align="center",
        colour=PURPLE,
        line_height=39,
    )

    axis_left, axis_right = panel_x + 270, panel_x + 690
    rows_y = [panel_y + 260, panel_y + 325, panel_y + 390, panel_y + 455]
    values = [
        ("median", offset["coordinate_abs_feature_px"]["median"], BLUE),
        ("mean", offset["coordinate_abs_feature_px"]["mean"], TEAL),
        ("p90", offset["coordinate_abs_feature_px"]["p90"], ORANGE),
        ("max", offset["coordinate_abs_feature_px"]["max"], RED),
    ]

    def log_x(value):
        return axis_left + math.log10(value) / 3.0 * (axis_right - axis_left)

    for tick in [1, 10, 100, 1000]:
        xx = log_x(tick)
        line(ctx, xx, panel_y + 215, xx, panel_y + 485, colour=GRID, width=2)
        draw_text(ctx, str(tick), xx, panel_y + 525, size=34, align="center", colour=MID)
    for yy, (label, value, colour) in zip(rows_y, values):
        draw_text(ctx, label, panel_x + 245, yy + 10, size=34, bold=True, align="right", colour=colour)
        line(ctx, axis_left, yy, log_x(value), yy, colour=colour, width=5)
        circle(ctx, log_x(value), yy, 12, fill=colour, stroke=WHITE, width=3)
        draw_text(ctx, f"{value:.2f}", log_x(value) - 8, yy - 20, size=34, align="right", colour=colour)

    mask = offset["mask"]
    box(
        ctx,
        panel_x + 65,
        panel_y + 535,
        305,
        130,
        f"Mask\nmean={mask['mean']:.3f}\nSD={mask['std']:.3f}",
        fill=LIGHT_GOLD,
        stroke=GOLD,
        size=34,
        bold=True,
        radius=14,
    )
    box(
        ctx,
        panel_x + 410,
        panel_y + 535,
        305,
        130,
        f"FAM(IR)\nspatial SD\n= {activation['spatial_std']:.0f}",
        fill=LIGHT_RED,
        stroke=RED,
        size=34,
        bold=True,
        radius=14,
    )
    surface.write_to_png(str(OUT / "fam_p5_seed41_collapse.png"))


def generate_yolo_architecture():
    """Render YOLOv10 as a portrait-native top-to-bottom pipeline."""
    width, height = 650, 910
    surface, ctx = pdf_canvas("yolov10_dual_backbone_fam.pdf", width, height)
    box(
        ctx,
        145,
        22,
        360,
        58,
        "Paired input: RGB (3 ch) + IR (1 ch)",
        fill=(0.96, 0.97, 0.98),
        stroke=NAVY,
        size=15,
        bold=True,
    )
    arrow(ctx, 325, 80, 325, 105, colour=NAVY)
    box(
        ctx,
        120,
        105,
        410,
        70,
        "Training-only feature gating\nVIS 20% | IR 20% | VIS+IR 60%\nAn absent backbone is not executed",
        fill=LIGHT_GOLD,
        stroke=GOLD,
        size=14,
        bold=True,
    )

    box(ctx, 30, 210, 270, 82, "RGB backbone\nYOLOv10-s, layers 0-10", fill=LIGHT_BLUE, stroke=BLUE, size=15, bold=True)
    box(ctx, 350, 210, 270, 82, "IR backbone\nYOLOv10-s, 1-channel", fill=LIGHT_ORANGE, stroke=ORANGE, size=15, bold=True)
    line(ctx, 325, 175, 325, 190, colour=GOLD, width=1.8)
    arrow(ctx, 325, 190, 165, 210, colour=BLUE, width=1.8)
    arrow(ctx, 325, 190, 485, 210, colour=ORANGE, width=1.8)

    levels = [
        ("P3", "L4 | 128 ch | s=8", 330),
        ("P4", "L6 | 256 ch | s=16", 415),
        ("P5", "L10 | 512 ch | s=32", 500),
    ]
    fused_centres = []
    for level, metadata, y in levels:
        box(ctx, 28, y, 125, 28, f"RGB {level}", fill=LIGHT_BLUE, stroke=BLUE, size=13, bold=True)
        box(ctx, 28, y + 38, 125, 28, f"IR {level}", fill=LIGHT_ORANGE, stroke=ORANGE, size=13, bold=True)
        draw_text(ctx, metadata, 172, y + 34, size=13, colour=MID, max_width=140)
        box(ctx, 325, y + 3, 190, 60, f"Additive or FAM\n{level} fusion", fill=LIGHT_TEAL, stroke=TEAL, size=14, bold=True)
        box(ctx, 555, y + 9, 80, 48, f"fused\n{level}", fill=LIGHT_TEAL, stroke=TEAL, size=13, bold=True)
        arrow(ctx, 153, y + 14, 325, y + 18, colour=BLUE, head=7)
        arrow(ctx, 153, y + 52, 325, y + 48, colour=ORANGE, head=7)
        arrow(ctx, 515, y + 33, 555, y + 33, colour=TEAL, head=7)
        fused_centres.append(y + 33)

    for centre_y in fused_centres:
        line(ctx, 635, centre_y, 642, centre_y, colour=TEAL, width=2)
    line(ctx, 642, fused_centres[0], 642, 620, colour=TEAL, width=2)
    arrow(ctx, 642, 620, 575, 620, colour=TEAL, width=2, head=8)
    box(
        ctx,
        75,
        610,
        500,
        75,
        "Shared FPN / PAN neck\nTop-down: P5 → P4 → P3 | Bottom-up: P3 → P4 → P5",
        fill=LIGHT_GREEN,
        stroke=GREEN,
        size=14,
        bold=True,
    )

    box(ctx, 120, 720, 180, 68, "v10Detect\none-to-many + one-to-one", fill=LIGHT_PURPLE, stroke=PURPLE, size=14, bold=True)
    box(ctx, 350, 720, 180, 68, "Person\ndetections", fill=LIGHT_TEAL, stroke=TEAL, size=14, bold=True)
    arrow(ctx, 325, 685, 210, 720, colour=GREEN, head=8)
    arrow(ctx, 300, 754, 350, 754, colour=PURPLE, head=8)

    box(ctx, 35, 805, 580, 95, "", fill=(0.975, 0.978, 0.982), stroke=NAVY, line_width=1.8)
    draw_text(ctx, "Standalone routes", 325, 827, size=15, bold=True, align="center", colour=NAVY)
    draw_text(ctx, "VIS+IR [1,1]: both backbones; FAM active", 65, 850, size=13, colour=TEAL, bold=True)
    draw_text(ctx, "VIS [1,0]: RGB only; FAM bypassed", 65, 870, size=13, colour=BLUE, bold=True)
    draw_text(ctx, "IR [0,1]: IR only; FAM bypassed", 65, 890, size=13, colour=ORANGE, bold=True)
    surface.finish()


def main():
    generate_timeline()
    generate_rtdetr_architecture()
    generate_rtdetr_results()
    generate_p5_collapse()
    generate_yolo_architecture()
    for path in sorted(OUT.glob("*.pdf")):
        if path.name in {
            "experimental_timeline.pdf",
            "rtdetr_fam_architecture.pdf",
            "rtdetr_final_paired_map50.pdf",
            "yolov10_dual_backbone_fam.pdf",
        }:
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
