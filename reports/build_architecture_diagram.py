"""Generates reports/figures/architecture.png -- the architecture diagram
referenced by README.md, docs/ARCHITECTURE.md, and the DOCX report.

Layout is a strict grid: every arrow connects two boxes that are either
in the same row (horizontal) or the same column (vertical) -- no
diagonal connectors -- so nothing crosses another box or another arrow.
The main pipeline snakes top-to-bottom (row 1 left-to-right, row 2
right-to-left, row 3 left-to-right again); MLflow and the two monitoring
boxes are short dead-end branches off the main spine, drawn in their own
colour so they read as side branches rather than a continuation.

Usage:
    python reports/build_architecture_diagram.py
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from heart_mlops.config import REPORTS_FIGURES_DIR

COLORS = {
    "data": "#4C72B0",
    "train": "#55A868",
    "track": "#C44E52",
    "cicd": "#8172B2",
    "serve": "#CCB974",
    "monitor": "#64B5CD",
}

BOX_W = 1.7
BOX_H = 0.95


def _box(ax, x, y, text, color, fontsize=9.0):
    box = mpatches.FancyBboxPatch(
        (x, y),
        BOX_W,
        BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3,
        edgecolor="#333333",
        facecolor=color,
        alpha=0.9,
    )
    ax.add_patch(box)
    ax.text(
        x + BOX_W / 2,
        y + BOX_H / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="white",
    )
    return (x, y)


def _arrow_right(ax, box_a, label=None):
    """Horizontal arrow from box_a's right edge to the next box (box_a's
    x + one grid pitch), same y -- used for same-row connections."""
    ax_, ay_ = box_a
    start = (ax_ + BOX_W, ay_ + BOX_H / 2)
    end = (ax_ + BOX_W + 0.35, ay_ + BOX_H / 2)
    _draw_arrow(ax, start, end, label)


def _arrow_left(ax, box_a, label=None):
    ax_, ay_ = box_a
    start = (ax_, ay_ + BOX_H / 2)
    end = (ax_ - 0.35, ay_ + BOX_H / 2)
    _draw_arrow(ax, start, end, label)


def _arrow_down(ax, box_a, label=None):
    """Vertical arrow from box_a's bottom edge straight down -- used when
    the next box in the same column is in the row below."""
    ax_, ay_ = box_a
    start = (ax_ + BOX_W / 2, ay_)
    end = (ax_ + BOX_W / 2, ay_ - 0.35)
    _draw_arrow(ax, start, end, label, label_dx=0.55)


def _draw_arrow(ax, start, end, label=None, label_dx=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.4,
        color="#555555",
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx + label_dx, my, label, ha="left", va="center", fontsize=7, color="#333333")


def build() -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(-0.3, 13.0)
    ax.set_ylim(0.8, 10.3)
    ax.axis("off")

    pitch = BOX_W + 0.35  # horizontal distance between adjacent box left-edges in a row
    row_pitch = BOX_H + 1.1  # vertical distance between row baselines

    row1_y = 9.0
    row2_y = row1_y - row_pitch
    row3_y = row2_y - row_pitch
    row4_y = row3_y - row_pitch

    # --- Row 1: data -> EDA -> features -> train, then branch to MLflow ---
    x = 0.2
    uci = _box(ax, x, row1_y, "UCI Heart\nDisease (4 sites)", COLORS["data"])
    _arrow_right(ax, uci)
    x += pitch
    clean = _box(ax, x, row1_y, "download.py\nclean.py", COLORS["data"])
    _arrow_right(ax, clean)
    x += pitch
    eda = _box(ax, x, row1_y, "EDA notebook\n+ figures", COLORS["data"])
    _arrow_right(ax, eda)
    x += pitch
    features = _box(ax, x, row1_y, "Feature pipeline\n(preprocess.py)", COLORS["train"])
    _arrow_right(ax, features)
    x += pitch
    train_x = x
    train = _box(ax, x, row1_y, "Tune + select\n3 models\n(train.py)", COLORS["train"])
    _arrow_right(ax, train)
    x += pitch
    _box(ax, x, row1_y, "MLflow tracking\n+ registry\n(champion alias)", COLORS["track"])

    # --- Row 1 -> Row 2: train also feeds the packaged model, straight down ---
    _arrow_down(ax, train, label="same fitted\npipeline (D5.2)")
    packaged = _box(
        ax, train_x, row2_y, "Packaged model\n(joblib +\nmetadata.json)", COLORS["track"]
    )

    # --- Row 2: packaged model -> CI/CD -> Docker, flowing right to left ---
    _arrow_left(ax, packaged, label="commit\ntriggers")
    ci_x = train_x - pitch
    ci = _box(ax, ci_x, row2_y, "CI/CD\n(lint/test/\ntrain/docker)", COLORS["cicd"])
    _arrow_left(ax, ci, label="build +\nsmoke test")
    docker_x = ci_x - pitch
    docker = _box(ax, docker_x, row2_y, "Docker image\n(heart-mlops-api)", COLORS["cicd"])

    # --- Row 2 -> Row 3: docker image -> ingress, straight down ---
    _arrow_down(ax, docker, label="minikube\nimage load")
    ingress = _box(
        ax, docker_x, row3_y, "Ingress /\nLoadBalancer\n(heart-api.local)", COLORS["serve"]
    )

    # --- Row 3: ingress -> API pods, left to right ---
    _arrow_right(ax, ingress)
    pods_x = docker_x + pitch
    pods = _box(ax, pods_x, row3_y, "API pods x2\n(FastAPI,\nHPA 2-5)", COLORS["serve"])

    # --- Row 3 -> Row 4: API pods -> Prometheus -> Grafana ---
    _arrow_down(ax, pods, label="/metrics")
    prometheus = _box(ax, pods_x, row4_y, "Prometheus\n(pod scrape)", COLORS["monitor"])
    _arrow_right(ax, prometheus)
    grafana_x = pods_x + pitch
    _box(ax, grafana_x, row4_y, "Grafana\n(provisioned\ndashboard)", COLORS["monitor"])

    legend_patches = [
        mpatches.Patch(color=COLORS["data"], label="Data & EDA (Phase 2)"),
        mpatches.Patch(color=COLORS["train"], label="Features & modelling (Phase 3)"),
        mpatches.Patch(color=COLORS["track"], label="Tracking & packaging (Phase 4-5)"),
        mpatches.Patch(color=COLORS["cicd"], label="CI/CD & container (Phase 6-7)"),
        mpatches.Patch(color=COLORS["serve"], label="Kubernetes serving (Phase 7)"),
        mpatches.Patch(color=COLORS["monitor"], label="Monitoring (Phase 7)"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )

    ax.set_title(
        "Heart Disease Risk Prediction — End-to-End Architecture",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    fig.tight_layout()
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_FIGURES_DIR / "architecture.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    build()
