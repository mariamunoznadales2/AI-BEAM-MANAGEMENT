"""
Step 9 - Publication figures for the IEEE paper (light style).

Rebuilds the 2D figures from the saved results (no ray tracing, no training):
  fig1_beam_maps         (a) best beam per location, (b) loss of the
                         geometric beam                    [double column]
  fig2_beam_prediction   (a) position-only prediction, (b) prediction with a
                         few measured beams (3GPP BM-Case 1) [double column]
  fig3_frequency         top-1 accuracy at 3.5 vs 4.7 GHz   [single column]
  fig4_walking_user      temporal prediction while walking [single column]

Sized for IEEE Transactions (single column 3.5 in, double column 7.16 in),
8-pt serif text, colour-blind-safe palette plus markers/dashes so the plots
also read in grayscale. Each figure is saved as PDF (vector, for LaTeX) and
PNG (300 dpi, for slides).

Run from ~/Desktop/uos-beam-twin with the environment active
(after steps 3-6 for both frequencies):
    python 09_paper_figures.py
Outputs: figures/paper/fig*.pdf and figures/paper/fig*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap    # noqa: E402
from matplotlib.patches import Polygon as MplPolygon     # noqa: E402

import uos_twin as uos                                   # noqa: E402

OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)
BAND = "3p5GHz"                     # band used for the single-band figures
BANDS = {"3p5GHz": "3.5 GHz", "4p7GHz": "4.7 GHz"}
COL1, COL2 = 3.5, 7.16              # IEEE column widths [in]

# ------------------------------------------------------------ style
INK, MUTED, GRID = "#1f1f1e", "#5f5f5a", "#e3e2de"
BUILD_FILL, BUILD_EDGE = "#d9d8d3", "#a9a8a2"
# Validated colour-blind-safe categorical palette (fixed order)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
DASHES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
BEAM_CMAP = LinearSegmentedColormap.from_list(
    "beams", ["#104281", "#3987e5", "#eeede9", "#e34948", "#9e1b1b"])
LOSS_CMAP = LinearSegmentedColormap.from_list(
    "loss", ["#eef5fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load(name: str, tag: str = BAND) -> dict:
    return json.loads(uos.tagged(f"results/{name}.json", tag).read_text())


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figures/paper/{name}.pdf / .png")


def panel_label(ax, text: str):
    ax.text(-0.02, 1.02, text, transform=ax.transAxes, fontweight="bold",
            ha="right", va="bottom", fontsize=8)


def curve(ax, xs, ys, k, label):
    ax.plot(xs, ys, color=C[k], marker=MARKERS[k], linestyle=DASHES[k],
            linewidth=1.3, markersize=3.6, markeredgecolor="white",
            markeredgewidth=0.6, label=label)


# ============================================================ Fig. 1
D = np.load(uos.tagged("data/beams_uos.npz", BAND))
angles, best, covered = D["angles"], D["best"], D["covered"]
g = D["gains_db"]
geo_loss = g.max(0) - np.take_along_axis(g, D["geo"][None], 0)[0]
X, Y, cell = D["x"], D["y"], float(D["cell"])
extent = (X[0, 0] - cell / 2, X[0, -1] + cell / 2,
          Y[0, 0] - cell / 2, Y[-1, 0] + cell / 2)
tx = D["tx_pos"]
R = uos.rotation_matrix_np(*D["tx_orientation"])
campus = uos.campus_polygon()
footprints = uos.building_footprints()
cx, cy = campus.exterior.xy


def map_axes(ax):
    for fp in footprints:
        ax.add_patch(MplPolygon(fp, closed=True, facecolor=BUILD_FILL,
                                edgecolor=BUILD_EDGE, linewidth=0.25, zorder=3))
    ax.plot(cx, cy, color=INK, linewidth=0.7, linestyle=(0, (3, 2)), zorder=4)
    ax.scatter([tx[0]], [tx[1]], marker="*", s=70, color=INK,
               edgecolor="white", linewidth=0.6, zorder=6)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x [m]")
    ax.tick_params(length=2)


fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.05), constrained_layout=True)
ax = axes[0]
im = ax.imshow(np.where(covered, angles[best], np.nan), origin="lower",
               extent=extent, cmap=BEAM_CMAP, vmin=angles[0], vmax=angles[-1],
               interpolation="nearest", zorder=2, rasterized=True)
for k in range(0, len(angles), 4):                   # beam directions
    ph = np.radians(angles[k])
    d = R @ np.array([np.cos(ph), np.sin(ph), 0.0])
    d = d[:2] / np.linalg.norm(d[:2])
    ax.plot([tx[0], tx[0] + 1000 * d[0]], [tx[1], tx[1] + 1000 * d[1]],
            color=MUTED, linewidth=0.35, alpha=0.7, zorder=5)
map_axes(ax)
ax.set_ylabel("y [m]")
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
cb.set_label("Best-beam steering angle [deg]")
cb.outline.set_visible(False)
panel_label(ax, "(a)")
ax.annotate("gNB", (tx[0], tx[1]), xytext=(6, -10), textcoords="offset points",
            ha="left", fontsize=7, zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

ax = axes[1]
im = ax.imshow(np.where(covered, np.minimum(geo_loss, 30), np.nan),
               origin="lower", extent=extent, cmap=LOSS_CMAP, vmin=0, vmax=30,
               interpolation="nearest", zorder=2, rasterized=True)
map_axes(ax)
ax.tick_params(labelleft=False)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01, extend="max")
cb.set_label("Loss of the geometric beam [dB]")
cb.outline.set_visible(False)
panel_label(ax, "(b)")
save(fig, "fig1_beam_maps")

# ============================================================ Fig. 2
m = load("metrics")
mb = load("metrics_setB")
fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.75))
fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.33, wspace=0.22)

ax = axes[0]
ks = np.arange(1, len(m["neural_net"]["loss_db"]) + 1)
for k, (key, label) in enumerate([("neural_net", "Neural network (position)"),
                                  ("knn", "kNN lookup (position)"),
                                  ("geometric", "Geometric beam")]):
    curve(ax, ks, m[key]["loss_db"], k, label)
ax.set_xticks(ks)
ax.set_xlabel("Beams tested after the prediction ($k$)")
ax.set_ylabel("Mean loss vs. exhaustive sweep [dB]")
ax.set_ylim(0, None)
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), ncol=2,
          handlelength=2.4, columnspacing=1.0, fontsize=6.5)
panel_label(ax, "(a)")

ax = axes[1]
series = [("position + Set B NN |B|=4", "Position + 4 measured (NN)", 4),
          ("Set B NN |B|=4", "4 measured (NN)", 4),
          ("Set B strongest |B|=4", "4 measured, strongest (no AI)", 4),
          ("position NN", "Position only (NN)", 0),
          ("geometric", "Geometric beam", 0)]
order = [0, 1, 2, 3, 4]
for k, (key, label, nb) in zip(order, series):
    cv = mb["curves"][key]
    xs = [b for b in range(1, len(cv) + 1) if cv[b - 1] is not None and b > nb]
    ys = [cv[b - 1] for b in xs]
    curve(ax, xs, ys, k, label)
ax.set_ylim(0, 6.5)
ax.set_xticks(range(1, len(mb["curves"]["geometric"]) + 1))
ax.set_xlabel("Total beams measured")
ax.set_ylabel("Mean loss vs. exhaustive sweep [dB]")
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), ncol=2,
          handlelength=2.4, columnspacing=1.0, fontsize=6.5)
ax.text(0.27, 0.9, f"Set B = 4 of 32 beams\n{mb['noise_db']:.0f} dB measurement noise",
        transform=ax.transAxes, fontsize=6.5, color=MUTED, va="top")
panel_label(ax, "(b)")
save(fig, "fig2_beam_prediction")

# ============================================================ Fig. 3
methods = [("Geometric\n(no AI)", lambda r, s: r["geometric"]["top1_acc"]),
           ("kNN", lambda r, s: r["knn"]["top1_acc"]),
           ("Position\nNN", lambda r, s: r["neural_net"]["top1_acc"]),
           ("Position +\n4 meas. NN", lambda r, s: s["top1"]["position + Set B NN |B|=4"]),
           ("8 meas.\nNN", lambda r, s: s["top1"]["Set B NN |B|=8"])]
fig, ax = plt.subplots(figsize=(COL1, 2.2), constrained_layout=True)
xpos = np.arange(len(methods))
w = 0.36
for j, (tag, lab) in enumerate(BANDS.items()):
    r, s = load("metrics", tag), load("metrics_setB", tag)
    vals = [100 * f(r, s) for _, f in methods]
    bars = ax.bar(xpos + (j - 0.5) * (w + 0.03), vals, width=w, color=C[j],
                  hatch=None if j == 0 else "////", edgecolor="white",
                  linewidth=0.4, label=lab, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}",
                ha="center", va="bottom", fontsize=6)
ax.set_xticks(xpos)
ax.set_xticklabels([n for n, _ in methods])
ax.set_ylabel("Top-1 accuracy [%]")
ax.set_ylim(0, 100)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", ncol=2)
save(fig, "fig3_frequency")

# ============================================================ Fig. 4
rows = [("full sweep, one step late", "loss_top1_db",
         "Full sweep,\n1 step late (32)"),
        ("Set B NN (current step only)", "loss_top3_db",
         "AI, current\nstep (4+3)"),
        ("GRU (last 4 steps)", "loss_top3_db", "AI, GRU\n4 steps (4+3)")]
fig, ax = plt.subplots(figsize=(COL1, 2.0), constrained_layout=True)
xpos = np.arange(len(rows))
for j, (tag, lab) in enumerate(BANDS.items()):
    t = load("metrics_temporal", tag)
    vals = [t[k][metric] for k, metric, _ in rows]
    bars = ax.bar(xpos + (j - 0.5) * (w + 0.03), vals, width=w, color=C[j],
                  hatch=None if j == 0 else "////", edgecolor="white",
                  linewidth=0.4, label=lab, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6)
ax.set_xticks(xpos)
ax.set_xticklabels([r[2] for r in rows])
ax.set_ylabel("Mean loss at next step [dB]")
ax.set_ylim(0, None)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", ncol=2)
save(fig, "fig4_walking_user")
