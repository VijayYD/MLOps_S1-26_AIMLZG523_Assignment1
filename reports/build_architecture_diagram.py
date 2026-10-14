"""Generates reports/figures/architecture.png -- the architecture diagram
referenced by README.md, docs/ARCHITECTURE.md, and the DOCX report.

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


def _box(ax, xy, w, h, text, color, fontsize=9.5):
    x, y = xy
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3,
        edgecolor="#333333",
        facecolor=color,
        alpha=0.85,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="white",
        wrap=True,
    )
    return (x, y, w, h)


def _arrow(ax, box_a, box_b, label=None, style="-|>"):
    ax_, ay_, aw, ah = box_a
    bx_, by_, bw, bh = box_b
    start = (ax_ + aw / 2, ay_ + ah / 2)
    end = (bx_ + bw / 2, by_ + bh / 2)
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        connectionstyle="arc3,rad=0.0",
        mutation_scale=16,
        linewidth=1.4,
        color="#555555",
        shrinkA=18,
        shrinkB=18,
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.15, label, ha="center", va="bottom", fontsize=7.5, color="#333333")


def build() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Row 1: data + EDA
    uci = _box(ax, (0.3, 6.3), 1.9, 1.0, "UCI Heart\nDisease\n(4 sites)", COLORS["data"])
    clean = _box(ax, (2.6, 6.3), 1.9, 1.0, "download.py\nclean.py", COLORS["data"])
    eda = _box(ax, (4.9, 6.3), 1.9, 1.0, "EDA notebook\n+ figures", COLORS["data"])

    # Row 1 cont: features + train
    features = _box(ax, (7.2, 6.3), 1.9, 1.0, "Feature\npipeline\n(preprocess.py)", COLORS["train"])
    train = _box(ax, (9.5, 6.3), 2.0, 1.0, "Tune + select\n3 models\n(train.py)", COLORS["train"])

    _arrow(ax, uci, clean)
    _arrow(ax, clean, eda)
    _arrow(ax, eda, features)
    _arrow(ax, features, train)

    # Row 2: MLflow + packaging
    mlflow_box = _box(
        ax, (9.5, 4.6), 2.0, 1.0, "MLflow tracking\n+ registry\n(champion alias)", COLORS["track"]
    )
    packaged = _box(
        ax, (7.2, 4.6), 1.9, 1.0, "Packaged model\n(joblib +\nmetadata.json)", COLORS["track"]
    )

    _arrow(ax, train, mlflow_box)
    _arrow(ax, train, packaged, label="same fitted\npipeline, D5.2")

    # Row 3: CI/CD + Docker
    ci = _box(ax, (9.5, 2.9), 2.0, 1.0, "CI/CD\n(lint/test/\ntrain/docker)", COLORS["cicd"])
    docker = _box(ax, (7.2, 2.9), 1.9, 1.0, "Docker image\n(heart-mlops-api)", COLORS["cicd"])

    _arrow(ax, packaged, ci, label="commits trigger")
    _arrow(ax, ci, docker, label="build + smoke test")

    # Row 4: Minikube serving
    ingress = _box(
        ax, (4.9, 1.2), 1.9, 1.0, "Ingress /\nLoadBalancer\n(heart-api.local)", COLORS["serve"]
    )
    pods = _box(ax, (2.6, 1.2), 1.9, 1.0, "API pods x2\n(FastAPI,\nHPA 2-5)", COLORS["serve"])

    _arrow(ax, docker, ingress, label="minikube image load")
    _arrow(ax, ingress, pods)

    # Monitoring branch
    prometheus = _box(ax, (0.3, 1.2), 1.9, 1.0, "Prometheus\n(pod scrape)", COLORS["monitor"])
    grafana = _box(ax, (0.3, 0.0), 1.9, 1.0, "Grafana\n(provisioned\ndashboard)", COLORS["monitor"])

    _arrow(ax, pods, prometheus, label="/metrics")
    _arrow(ax, prometheus, grafana)

    # Legend
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
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )

    ax.set_title(
        "Heart Disease Risk Prediction — End-to-End Architecture",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    fig.tight_layout()
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_FIGURES_DIR / "architecture.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    build()
