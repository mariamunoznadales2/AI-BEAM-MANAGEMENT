"""
Step 4 - Train a neural network that predicts the best beam from the
user's position (no beam sweep needed).

Input : user position (x, y, z) in the campus digital twin
Output: probability of each of the 32 beams being the best one

The network is compared with two baselines:
  - geometric beam : point straight at the user (no learning)
  - kNN lookup     : copy the best beam of the nearest training positions
and with the exhaustive sweep of all 32 beams (upper bound, 0 dB loss).

Data is split by 40 m x 40 m spatial blocks, so test positions are areas the
model has never seen (a random split would leak neighbouring cells).

Run from ~/Desktop/uos-beam-twin with the environment active:
    python 04_train_beam_predictor.py              (3.5 GHz, default)
    python 04_train_beam_predictor.py --freq 4.7   (4.7 GHz)
Outputs (<freq> = 3p5GHz or 4p7GHz):
  models/beam_mlp_<freq>.pt                trained model
  results/metrics_<freq>.json              test metrics
  figures/power_loss_vs_beams_<freq>.png   power loss vs. number of beams measured
  figures/prediction_vs_truth_<freq>.png   ray-traced vs. predicted best beam
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402
import torch.nn as nn                                    # noqa: E402
import torch.nn.functional as F                          # noqa: E402
from matplotlib.colors import LinearSegmentedColormap    # noqa: E402
from matplotlib.patches import Polygon as MplPolygon     # noqa: E402
from matplotlib.patches import Rectangle                 # noqa: E402
from scipy.spatial import KDTree                         # noqa: E402

import uos_twin as uos                                   # noqa: E402

SEED = 0
BLOCK = 40.0          # spatial block size for the train/val/test split [m]
SPLIT = (0.70, 0.15, 0.15)
N_FREQ = 3            # Fourier-feature frequencies per coordinate
HIDDEN = 256
DROPOUT = 0.1
WEIGHT_DECAY = 1e-3
EPOCHS = 150
BATCH = 1024
LR = 2e-3
TEMP_DB = 2.0         # soft-label temperature [dB]
K_NN = 5
MAX_K = 8             # beams measured after the prediction (top-k)
FIG_DIR, MODEL_DIR, RES_DIR = Path("figures"), Path("models"), Path("results")
for p in (FIG_DIR, MODEL_DIR, RES_DIR):
    p.mkdir(exist_ok=True)

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Training device:", device)

# ------------------------------------------------------------ data
D = np.load(uos.tagged("data/beams_uos.npz"))
print(f"Carrier frequency: {uos.FREQ_GHZ:g} GHz")
valid = D["covered"] & D["in_sector"]
pos = np.stack([D["x"][valid], D["y"][valid], D["z"][valid]], 1)
gains = D["gains_db"][:, valid].T                  # (N, 32) in dB
best = D["best"][valid]
geo = D["geo"][valid]
n_beams = gains.shape[1]
print(f"Samples: {len(pos)}  |  beams: {n_beams}")

# Spatial block split
bx = np.floor(pos[:, 0] / BLOCK).astype(int)
by = np.floor(pos[:, 1] / BLOCK).astype(int)
blocks = np.unique(np.stack([bx, by], 1), axis=0)
rng.shuffle(blocks)
n_tr = int(SPLIT[0] * len(blocks))
n_va = int(SPLIT[1] * len(blocks))
block_id = {tuple(b): (0 if i < n_tr else 1 if i < n_tr + n_va else 2)
            for i, b in enumerate(blocks)}
part = np.array([block_id[(a, b)] for a, b in zip(bx, by)])
tr, va, te = part == 0, part == 1, part == 2
print(f"Train {tr.sum()}  |  val {va.sum()}  |  test {te.sum()} samples")

# Normalise positions to [-1, 1] with training statistics
lo, hi = pos[tr].min(0), pos[tr].max(0)
pos_n = 2 * (pos - lo) / (hi - lo) - 1

# Soft labels: beams within a few dB of the best one also get probability
soft = torch.softmax(torch.tensor((gains - gains.max(1, keepdims=True))
                                  / TEMP_DB, dtype=torch.float32), dim=1)


# ------------------------------------------------------------ model
class FourierMLP(nn.Module):
    """MLP on Fourier features [sin(2^k pi p), cos(2^k pi p)] of the position.
    The features let a small network represent the sharp beam boundaries
    created by buildings (a plain MLP on (x, y, z) is too smooth)."""

    def __init__(self, n_freq: int, hidden: int, n_out: int):
        super().__init__()
        self.freqs: torch.Tensor
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * torch.pi)
        n_in = 3 + 3 * 2 * n_freq
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_out))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        ang = p[..., None] * self.freqs                # (B, 3, F)
        feats = torch.cat([p, torch.sin(ang).flatten(1),
                           torch.cos(ang).flatten(1)], dim=1)
        return self.net(feats)


model = FourierMLP(N_FREQ, HIDDEN, n_beams).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

X = torch.tensor(pos_n, dtype=torch.float32, device=device)
Y = soft.to(device)
idx_tr = torch.tensor(np.flatnonzero(tr), device=device)
idx_va = torch.tensor(np.flatnonzero(va), device=device)


def val_loss() -> float:
    model.eval()
    with torch.no_grad():
        logp = F.log_softmax(model(X[idx_va]), dim=1)
        return float(-(Y[idx_va] * logp).sum(1).mean())


# ------------------------------------------------------------ training
t0 = time.time()
best_val, best_state = float("inf"), None
loss = torch.tensor(0.0)
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = idx_tr[torch.randperm(len(idx_tr), device=device)]
    for i in range(0, len(perm), BATCH):
        b = perm[i:i + BATCH]
        logp = F.log_softmax(model(X[b]), dim=1)
        loss = -(Y[b] * logp).sum(1).mean()       # cross-entropy, soft labels
        opt.zero_grad()
        loss.backward()
        opt.step()
    sched.step()
    v = val_loss()
    if v < best_val:
        best_val = v
        best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    if epoch % 50 == 0:
        print(f"  epoch {epoch:3d}  train {loss.item():.3f}  val {v:.3f}")
print(f"Training finished in {time.time() - t0:.1f} s")
assert best_state is not None
model.load_state_dict(best_state)
torch.save({"state_dict": best_state, "lo": lo, "hi": hi, "n_freq": N_FREQ,
            "hidden": HIDDEN, "n_beams": n_beams}, uos.tagged("models/beam_mlp.pt"))

# ------------------------------------------------------------ predictions
model.eval()
with torch.no_grad():
    nn_scores = model(X).cpu().numpy()                 # (N, 32)

# kNN baseline: average soft labels of the K nearest training positions
tree = KDTree(pos[tr])
_, nb = tree.query(pos, k=K_NN)
knn_scores = soft.numpy()[np.flatnonzero(tr)][nb].mean(1)

# Geometric baseline: rank beams by angular distance to the user direction
beam_idx = np.arange(n_beams)
geo_scores = -np.abs(beam_idx[None, :] - geo[:, None]).astype(float)


def evaluate(scores: np.ndarray, mask: np.ndarray) -> dict:
    """Top-k accuracy and power loss when measuring only the top-k beams."""
    order = np.argsort(-scores[mask], axis=1)
    g, b = gains[mask], best[mask]
    out = {"top1_acc": float((order[:, 0] == b).mean()),
           "top3_acc": float((order[:, :3] == b[:, None]).any(1).mean()),
           "loss_db": []}
    for k in range(1, MAX_K + 1):
        picked = np.take_along_axis(g, order[:, :k], 1).max(1)
        out["loss_db"].append(float((g.max(1) - picked).mean()))
    return out


results = {"neural_net": evaluate(nn_scores, te),
           "knn": evaluate(knn_scores, te),
           "geometric": evaluate(geo_scores, te),
           "n_test": int(te.sum()), "n_beams": int(n_beams)}
with open(uos.tagged("results/metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nTest results (unseen areas)")
print(f"{'method':<12}{'top-1':>8}{'top-3':>8}{'loss k=1':>10}{'loss k=3':>10}")
for name in ("geometric", "knn", "neural_net"):
    r = results[name]
    print(f"{name:<12}{r['top1_acc'] * 100:7.1f}%{r['top3_acc'] * 100:7.1f}%"
          f"{r['loss_db'][0]:9.2f} dB{r['loss_db'][2]:7.2f} dB")
nn_loss = np.array(results["neural_net"]["loss_db"])
k_1db = int(np.argmax(nn_loss < 1.0)) + 1 if (nn_loss < 1.0).any() else None
if k_1db:
    print(f"Neural net needs {k_1db} of {n_beams} beams for < 1 dB loss "
          f"-> {100 * (1 - k_1db / n_beams):.0f} % less beam-sweep overhead")

# ------------------------------------------------------------ figures
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6b66", "#e6e5e1"
SERIES = {"neural_net": ("Neural network", "#2a78d6"),
          "knn": (f"kNN lookup (k={K_NN})", "#eb6834"),
          "geometric": ("Geometric beam", "#1baf7a")}

# Power loss vs. number of beams measured
fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ks = np.arange(1, MAX_K + 1)
for name, (label, color) in SERIES.items():
    y = results[name]["loss_db"]
    ax.plot(ks, y, color=color, linewidth=2, solid_capstyle="round",
            marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=2, label=label)
    ax.annotate(label, (ks[-1], y[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9)
ax.axhline(0, color=MUTED, linewidth=1)
ax.text(1, 0.15, f"Exhaustive sweep ({n_beams} beams) = 0 dB", color=MUTED,
        fontsize=9)
ax.set_xticks(ks)
ax.set_xlim(0.7, MAX_K + 2.6)
ax.set_xlabel("Beams measured after the prediction (top-k)", color=MUTED)
ax.set_ylabel("Mean power loss vs. best beam [dB]", color=MUTED)
ax.set_title(f"Beam prediction on unseen campus areas ({uos.FREQ_GHZ:g} GHz)", color=INK,
             loc="left", fontsize=13)
ax.grid(axis="y", color=GRID, linewidth=1)
ax.tick_params(colors=MUTED)
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")
fig.savefig(uos.tagged("figures/power_loss_vs_beams.png"), dpi=200,
            bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
print("Saved", uos.tagged("figures/power_loss_vs_beams.png"))

# Ray-traced vs. predicted best beam (whole area; test blocks outlined)
angles = D["angles"]
shape = D["best"].shape
truth = np.full(shape, np.nan)
pred = np.full(shape, np.nan)
truth[valid] = angles[best]
pred[valid] = angles[nn_scores.argmax(1)]
xs_c, ys_c = D["x"][0], D["y"][:, 0]
half = float(D["cell"]) / 2
extent = (xs_c[0] - half, xs_c[-1] + half, ys_c[0] - half, ys_c[-1] + half)
beam_cmap = LinearSegmentedColormap.from_list(
    "beams", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9e1b1b"])
footprints = uos.building_footprints()
tx_pos = D["tx_pos"]
test_blocks = [b for b in blocks if block_id[tuple(b)] == 2]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURFACE)
im = None
for ax, img, title in zip(axes, (truth, pred),
                          ("Ray tracing (exhaustive sweep)",
                           "Neural network (position only)")):
    ax.set_facecolor(SURFACE)
    im = ax.imshow(img, origin="lower", extent=extent, cmap=beam_cmap,
                   vmin=angles[0], vmax=angles[-1], interpolation="nearest")
    for fp in footprints:
        ax.add_patch(MplPolygon(fp, closed=True, facecolor="#dcdad4",
                                edgecolor="#b9b7b0", linewidth=0.4))
    for bx0, by0 in test_blocks:
        ax.add_patch(Rectangle((bx0 * BLOCK, by0 * BLOCK), BLOCK, BLOCK,
                                   fill=False, edgecolor=INK, linewidth=0.5,
                                   alpha=0.5))
    ax.scatter([tx_pos[0]], [tx_pos[1]], marker="*", s=220, color=INK,
               edgecolor=SURFACE, linewidth=1.5, zorder=5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title(title, color=INK, loc="left", fontsize=13)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
assert im is not None
cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
cb.set_label("Best beam direction [deg]", color=MUTED)
cb.ax.tick_params(colors=MUTED)
cb.outline.set_visible(False)
fig.text(0.01, 0.01, "Outlined squares: test areas never seen in training.",
         color=MUTED, fontsize=9)
fig.savefig(uos.tagged("figures/prediction_vs_truth.png"), dpi=200,
            bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
print("Saved", uos.tagged("figures/prediction_vs_truth.png"))
