"""Render histograms and FPR-vs-recall curves for Stage B score distributions."""
from __future__ import annotations
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def ascii_histogram(scores: list[float], bins: int = 15, range_min: float = -0.2, range_max: float = 0.8, width: int = 40) -> list[str]:
    bin_width = (range_max - range_min) / bins
    counts = [0] * bins
    for s in scores:
        if s < range_min:
            idx = 0
        elif s >= range_max:
            idx = bins - 1
        else:
            idx = int((s - range_min) / bin_width)
        counts[min(bins - 1, max(0, idx))] += 1

    max_count = max(counts) if counts else 1
    lines = []
    for i in range(bins):
        b_start = range_min + i * bin_width
        b_end = b_start + bin_width
        bar_len = int((counts[i] / max_count) * width)
        bar = "#" * bar_len
        lines.append(f"[{b_start:+.2f}, {b_end:+.2f}) | {counts[i]:>4} | {bar}")
    return lines


def render_ascii_roc_curve(points: list[dict], width: int = 50, height: int = 15) -> str:
    """Render an ASCII curve of Recall vs FPR."""
    grid = [[" " for _ in range(width + 1)] for _ in range(height + 1)]

    # Draw axes
    for y in range(height + 1):
        grid[y][0] = "|"
    for x in range(width + 1):
        grid[height][x] = "-"
    grid[height][0] = "+"

    # Plot points
    for pt in points:
        fpr = pt["fpr"]
        rec = pt["recall"]
        x = int(round(fpr * width))
        y = height - int(round(rec * height))
        x = min(width, max(0, x))
        y = min(height, max(0, y))
        grid[y][x] = "*"

    # Mark tau = 0.13 point if available
    tau_point = next((p for p in points if abs(p["tau"] - 0.13) < 0.005), None)
    if tau_point:
        x = int(round(tau_point["fpr"] * width))
        y = height - int(round(tau_point["recall"] * height))
        x = min(width, max(0, x))
        y = min(height, max(0, y))
        grid[y][x] = "O"

    rows = []
    for y in range(height + 1):
        label = f"{1.0 - y / height:4.1f} " if y % 5 == 0 else "     "
        rows.append(label + "".join(grid[y]))
    x_axis_labels = "      0.0" + " " * (width - 10) + "1.0 (FPR)"
    rows.append(x_axis_labels)
    return "\n".join(rows)


def generate_svg_histogram(benign_scores: list[float], attack_scores: list[float], tau: float = 0.13) -> str:
    """Generate a clean SVG histogram of benign vs attack candidate similarity scores."""
    svg_w, svg_h = 700, 360
    margin = 50
    plot_w = svg_w - 2 * margin
    plot_h = svg_h - 2 * margin

    bins = 25
    r_min, r_max = -0.2, 0.8
    bw = (r_max - r_min) / bins

    b_counts = [0] * bins
    a_counts = [0] * bins
    for s in benign_scores:
        idx = min(bins - 1, max(0, int((s - r_min) / bw)))
        b_counts[idx] += 1
    for s in attack_scores:
        idx = min(bins - 1, max(0, int((s - r_min) / bw)))
        a_counts[idx] += 1

    max_c = max(max(b_counts), max(a_counts), 1)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}" style="background:#ffffff; font-family:sans-serif;">',
        f'<text x="{svg_w//2}" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#111827">Stage B Candidate Score Distributions (Validation Split)</text>',
        f'<rect x="{margin}" y="{margin}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#e5e7eb" stroke-width="1"/>',
    ]

    # Gridlines
    for step in range(5):
        y_val = margin + plot_h - (step / 4.0) * plot_h
        count_val = int((step / 4.0) * max_c)
        elements.append(f'<line x1="{margin}" y1="{y_val}" x2="{margin + plot_w}" y2="{y_val}" stroke="#e5e7eb" stroke-dasharray="3,3"/>')
        elements.append(f'<text x="{margin - 8}" y="{y_val + 4}" text-anchor="end" font-size="10" fill="#6b7280">{count_val}</text>')

    bar_step = plot_w / bins
    bar_w = bar_step * 0.42

    for i in range(bins):
        bx = margin + i * bar_step
        b_h = (b_counts[i] / max_c) * plot_h
        a_h = (a_counts[i] / max_c) * plot_h

        # Benign bar (blue)
        elements.append(f'<rect x="{bx + 2}" y="{margin + plot_h - b_h}" width="{bar_w}" height="{b_h}" fill="#3b82f6" opacity="0.8"/>')
        # Injected bar (red)
        elements.append(f'<rect x="{bx + 2 + bar_w}" y="{margin + plot_h - a_h}" width="{bar_w}" height="{a_h}" fill="#ef4444" opacity="0.8"/>')

        if i % 5 == 0:
            score_val = r_min + i * bw
            elements.append(f'<text x="{bx}" y="{margin + plot_h + 16}" text-anchor="middle" font-size="10" fill="#6b7280">{score_val:+.1f}</text>')

    # Tau vertical line
    tau_x = margin + ((tau - r_min) / (r_max - r_min)) * plot_w
    elements.append(f'<line x1="{tau_x}" y1="{margin}" x2="{tau_x}" y2="{margin + plot_h}" stroke="#10b981" stroke-width="2.5" stroke-dasharray="4,4"/>')
    elements.append(f'<text x="{tau_x + 5}" y="{margin + 20}" font-size="12" font-weight="bold" fill="#047857">tau = {tau:.2f}</text>')

    # Legend
    leg_x = svg_w - margin - 220
    elements.append(f'<rect x="{leg_x}" y="38" width="12" height="12" fill="#3b82f6"/>')
    elements.append(f'<text x="{leg_x + 18}" y="48" font-size="11" fill="#374151">Benign Candidates (N={len(benign_scores)})</text>')
    elements.append(f'<rect x="{leg_x}" y="56" width="12" height="12" fill="#ef4444"/>')
    elements.append(f'<text x="{leg_x + 18}" y="66" font-size="11" fill="#374151">Injected Candidates (N={len(attack_scores)})</text>')

    elements.append('</svg>')
    return "\n".join(elements)


def generate_svg_fpr_recall(points: list[dict]) -> str:
    """Render a validation FPR-versus-recall curve from a saved sweep."""
    width, height, margin = 560, 420, 55
    plot_w, plot_h = width - 2 * margin, height - 2 * margin
    coords = [
        (margin + float(point["fpr"]) * plot_w,
         margin + (1.0 - float(point["recall"])) * plot_h)
        for point in points
    ]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" style="background:#fff;font-family:sans-serif">',
        '<text x="280" y="24" text-anchor="middle" font-size="15" font-weight="bold">'
        'B-sem validation FPR vs recall</text>',
        f'<rect x="{margin}" y="{margin}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#9ca3af"/>',
    ]
    for tick in range(5):
        x = margin + tick * plot_w / 4
        y = margin + tick * plot_h / 4
        label = f"{tick / 4:.2f}"
        elements.extend([
            f'<line x1="{x}" y1="{margin}" x2="{x}" y2="{margin + plot_h}" stroke="#e5e7eb"/>',
            f'<line x1="{margin}" y1="{y}" x2="{margin + plot_w}" y2="{y}" stroke="#e5e7eb"/>',
            f'<text x="{x}" y="{margin + plot_h + 18}" text-anchor="middle" font-size="10">{label}</text>',
            f'<text x="{margin - 8}" y="{y + 3}" text-anchor="end" font-size="10">{1 - tick / 4:.2f}</text>',
        ])
    point_list = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    elements.append(f'<polyline points="{point_list}" fill="none" stroke="#2563eb" stroke-width="2"/>')
    for x, y in coords:
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="#1d4ed8"/>')
    elements.extend([
        f'<text x="{margin + plot_w / 2}" y="{height - 8}" text-anchor="middle" font-size="12">False positive rate</text>',
        f'<text transform="translate(15 {margin + plot_h / 2}) rotate(-90)" text-anchor="middle" font-size="12">Recall</text>',
        "</svg>",
    ])
    return "\n".join(elements)


def main():
    dist_file = RESULTS / "stage_b_score_distributions.json"
    ablation_file = RESULTS / "stage_b_ablation.json"

    if not dist_file.exists() or not ablation_file.exists():
        print("Required results files missing. Run 'python scripts/stage_b_ablation.py' first.")
        sys.exit(1)

    with open(dist_file, "r", encoding="utf-8") as f:
        distributions = json.load(f)
    with open(ablation_file, "r", encoding="utf-8") as f:
        ablation = json.load(f)

    print("=" * 80)
    print("STAGE B SCORE DISTRIBUTIONS & FPR-VS-RECALL CURVES")
    print("=" * 80)

    for setting in ["base", "enhanced"]:
        scores = distributions[setting]
        benign_scores = scores["benign"]
        attack_scores = scores["injected"]

        print(f"\n--- Setting: {setting.upper()} ---")
        print(f"Benign candidates count: {len(benign_scores)}")
        print(f"Injected candidates count: {len(attack_scores)}")

        print("\n[Benign Candidates Histogram]")
        for line in ascii_histogram(benign_scores, bins=10):
            print("  " + line)

        print("\n[Injected Candidates Histogram]")
        for line in ascii_histogram(attack_scores, bins=10):
            print("  " + line)

        setting_res = ablation["results"][setting]
        sem_points = setting_res["b_sem_points"]
        print("\n[B-sem Alone: FPR-vs-Recall Curve ('O' marks tau=0.13)]")
        print(render_ascii_roc_curve(sem_points))

        # Save SVG histogram
        svg_content = generate_svg_histogram(benign_scores, attack_scores, tau=0.13)
        svg_path = RESULTS / f"stage_b_histogram_{setting}.svg"
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        print(f"Saved SVG histogram to: {svg_path}")
        curve_path = RESULTS / f"stage_b_fpr_recall_{setting}.svg"
        curve_path.write_text(generate_svg_fpr_recall(sem_points), encoding="utf-8")
        print(f"Saved SVG FPR-recall curve to: {curve_path}")

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
