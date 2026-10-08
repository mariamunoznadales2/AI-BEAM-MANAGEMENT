"""
Step 8 - Compare 3.5 GHz (public 5G, band n78) with 4.7 GHz (Korean private
5G "e-Um 5G", band n79).

Reads the results of steps 3-6 for both frequencies and summarises them in
one table and one figure. Run steps 2-6 with --freq 4.7 first, e.g.:
    python 02_campus_coverage.py --freq 4.7
    python 03_beam_map.py --freq 4.7
    python 04_train_beam_predictor.py --freq 4.7
    python 05_beam_prediction_setB.py --freq 4.7
    python 06_walking_user.py --freq 4.7
then:
    python 08_compare_frequencies.py
Outputs:
  results/frequency_comparison.json
  figures/frequency_comparison.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
from matplotlib.patches import Rectangle                 # noqa: E402

import uos_twin as uos                                   # noqa: E402

BANDS = {"3p5GHz": "3.5 GHz (public 5G)", "4p7GHz": "4.7 GHz (private 5G)"}
COLORS = {"3p5GHz": "#2a78d6", "4p7GHz": "#eb6834"}


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


summary: dict[str, dict[str, float | None]] = {}
for tag in BANDS:
    npz = uos.tagged("data/beams_uos.npz", tag)
    if not npz.exists():
        print(f"Missing {npz}: run 03_beam_map.py for this frequency first")
        continue
    D = np.load(npz)
    g = D["gains_db"]
    covered, in_sector = D["covered"], D["in_sector"]
    ok = covered & in_sector
    best, geo = D["best"], D["geo"]
    geo_loss = (g.max(0) - np.take_along_axis(g, geo[None], 0)[0])[ok]
    m4 = load_json(uos.tagged("results/metrics.json", tag))
    m5 = load_json(uos.tagged("results/metrics_setB.json", tag))
    m6 = load_json(uos.tagged("results/metrics_temporal.json", tag))
    summary[tag] = {
        "coverage_pct": float(covered.mean() * 100),
        "mean_best_gain_db": float(g.max(0)[ok].mean()),
        "geometric_mismatch_pct": float(((np.abs(best - geo) > 1) & ok).sum()
                                        / ok.sum() * 100),
        "geometric_loss_db": float(geo_loss.mean()),
        "position_nn_top1_pct": (m4["neural_net"]["top1_acc"] * 100) if m4 else None,
        "position_nn_loss_k4_db": m4["neural_net"]["loss_db"][3] if m4 else None,
        "pos_setB4_top1_pct": (m5["top1"]["position + Set B NN |B|=4"] * 100)
        if m5 else None,
        "pos_setB4_loss_8beams_db": m5["curves"]["position + Set B NN |B|=4"][7]
        if m5 else None,
        "gru_loss_top3_db": m6["GRU (last 4 steps)"]["loss_top3_db"] if m6 else None,
    }

Path("results").mkdir(exist_ok=True)
Path("results/frequency_comparison.json").write_text(json.dumps(summary, indent=2))

# ------------------------------------------------------------ table
ROWS = [("coverage_pct", "Coverage (> -130 dB) [%]"),
        ("mean_best_gain_db", "Mean best-beam path gain [dB]"),
        ("geometric_mismatch_pct", "Best beam != geometric beam [%]"),
        ("geometric_loss_db", "Loss of geometric beam [dB]"),
        ("position_nn_top1_pct", "Position NN top-1 [%]"),
        ("position_nn_loss_k4_db", "Position NN loss, 4 beams [dB]"),
        ("pos_setB4_top1_pct", "Position + 4 measured, top-1 [%]"),
        ("pos_setB4_loss_8beams_db", "Position + 4 measured, loss @ 8 beams [dB]"),
        ("gru_loss_top3_db", "Walking user (GRU), loss 7 beams/step [dB]")]
tags = list(summary)
print(f"\n{'':<44}" + "".join(f"{BANDS[t]:>24}" for t in tags))
for key, label in ROWS:
    vals = [summary[t][key] for t in tags]
    print(f"{label:<44}" + "".join(
        f"{'n/a':>24}" if v is None else f"{v:>24.2f}" for v in vals))
print("\nSaved results/frequency_comparison.json")

# ------------------------------------------------------------ figure
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6b66", "#e6e5e1"
PANELS = [("coverage_pct", "Coverage [%]"),
          ("geometric_mismatch_pct", "Best beam ≠ geometric [%]"),
          ("position_nn_top1_pct", "Position NN top-1 [%]"),
          ("pos_setB4_loss_8beams_db", "Loss, pos. + 4 meas. @ 8 beams [dB]")]
fig, axes = plt.subplots(1, len(PANELS), figsize=(12, 3.4), facecolor=SURFACE)
for ax, (key, title) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    vals = [summary[t][key] for t in tags]
    for i, (t, v) in enumerate(zip(tags, vals)):
        if v is None:
            continue
        ax.bar(i, v, width=0.55, color=COLORS[t], label=BANDS[t])
        ax.text(i, v, f"{v:.1f}" if v >= 1 else f"{v:.2f}", ha="center",
                va="bottom", color=INK, fontsize=9)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([BANDS[t].split(" (")[0] for t in tags], color=MUTED)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
handles = [Rectangle((0, 0), 1, 1, color=COLORS[t]) for t in tags]
fig.legend(handles, [BANDS[t] for t in tags], loc="upper right", ncol=2,
           frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle("UOS digital twin: 3.5 GHz vs 4.7 GHz", x=0.01, ha="left",
             color=INK, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig("figures/frequency_comparison.png", dpi=200, facecolor=SURFACE)
print("Saved figures/frequency_comparison.png")
