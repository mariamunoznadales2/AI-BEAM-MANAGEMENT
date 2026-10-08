"""
Step 5 - 3GPP-style beam prediction (BM-Case 1): measure a few beams (Set B)
and predict the best beam of the full codebook (Set A = 32 beams).

In 3GPP Rel-18/19 (TR 38.843) the UE measures a small Set B of beams and an
AI model predicts the best beam in the large Set A. Here Set B is a uniform
subset of the codebook (every 8th beam -> 4 beams, or every 4th -> 8 beams),
so no new ray tracing is needed: the measurements come from the step-3
dataset (data/beams_uos_<freq>.npz).
A 1 dB Gaussian measurement error is added to every Set B measurement.

Methods compared (same spatial train/val/test split as step 4):
  geometric           point at the user, then its angular neighbours
  position NN         step-4 network (position only)
  Set B strongest     strongest measured beam, then its neighbours (no AI)
  Set B NN            network fed with the Set B measurements
  position + Set B NN network fed with position and Set B measurements

All methods are compared at the same overhead: total beams measured =
|Set B| + extra beams tested from the ranked prediction.

Run from ~/Desktop/uos-beam-twin with the environment active:
    python 05_beam_prediction_setB.py              (3.5 GHz, default)
    python 05_beam_prediction_setB.py --freq 4.7   (4.7 GHz)
Outputs (<freq> = 3p5GHz or 4p7GHz):
  results/metrics_setB_<freq>.json
  figures/power_loss_vs_overhead_<freq>.png
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

import uos_twin as uos                                   # noqa: E402

SEED = 0
BLOCK = 40.0          # same spatial split as step 4
SPLIT = (0.70, 0.15, 0.15)
SET_B_STEPS = (8, 4)  # every 8th beam (|B|=4) and every 4th beam (|B|=8)
NOISE_DB = 1.0        # measurement error on Set B [dB]
FLOOR_DB = -140.0     # measurements below this are reported as the floor
N_FREQ = 3
HIDDEN = 256
DROPOUT = 0.1
WEIGHT_DECAY = 1e-3
EPOCHS = 150
BATCH = 1024
LR = 2e-3
TEMP_DB = 2.0
MAX_BUDGET = 12       # max total beams measured in the curves
FIG_DIR, RES_DIR = Path("figures"), Path("results")
for p in (FIG_DIR, RES_DIR):
    p.mkdir(exist_ok=True)

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
N, n_beams = gains.shape
print(f"Samples: {N}  |  Set A: {n_beams} beams")

# Same spatial block split as step 4 (same seed and block size)
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

lo, hi = pos[tr].min(0), pos[tr].max(0)
pos_n = 2 * (pos - lo) / (hi - lo) - 1
soft = torch.softmax(torch.tensor((gains - gains.max(1, keepdims=True))
                                  / TEMP_DB, dtype=torch.float32), dim=1)
noise_rng = np.random.default_rng(SEED + 1)


def measure(set_b: np.ndarray) -> np.ndarray:
    """Noisy Set B measurements [dB], one fresh noise draw per call."""
    m = gains[:, set_b] + NOISE_DB * noise_rng.standard_normal((N, len(set_b)))
    return np.maximum(m, FLOOR_DB)


def meas_features(m: np.ndarray) -> np.ndarray:
    """Absolute level and shape (relative to the strongest Set B beam)."""
    return np.concatenate([(m + 100.0) / 20.0,
                           (m - m.max(1, keepdims=True)) / 10.0], axis=1)


# ------------------------------------------------------------ model
class BeamNet(nn.Module):
    """MLP on [Fourier features of the position, measurement features]."""

    def __init__(self, use_pos: bool, n_extra: int):
        super().__init__()
        self.use_pos = use_pos
        self.freqs: torch.Tensor
        self.register_buffer("freqs", (2.0 ** torch.arange(N_FREQ)) * torch.pi)
        n_in = (3 + 6 * N_FREQ if use_pos else 0) + n_extra
        self.net = nn.Sequential(
            nn.Linear(n_in, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, n_beams))

    def forward(self, p: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.use_pos:
            ang = p[..., None] * self.freqs
            parts += [p, torch.sin(ang).flatten(1), torch.cos(ang).flatten(1)]
        parts.append(extra)
        return self.net(torch.cat(parts, dim=1))


def train_net(use_pos: bool, feats_train: np.ndarray,
              feats_eval: np.ndarray) -> np.ndarray:
    """Train on (train, val) with feats_train; return scores on feats_eval."""
    torch.manual_seed(SEED)
    model = BeamNet(use_pos, feats_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    P = torch.tensor(pos_n, dtype=torch.float32, device=device)
    E = torch.tensor(feats_train, dtype=torch.float32, device=device)
    Y = soft.to(device)
    i_tr = torch.tensor(np.flatnonzero(tr), device=device)
    i_va = torch.tensor(np.flatnonzero(va), device=device)
    best_val, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        perm = i_tr[torch.randperm(len(i_tr), device=device)]
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss = -(Y[b] * F.log_softmax(model(P[b], E[b]), 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = float(-(Y[i_va] * F.log_softmax(model(P[i_va], E[i_va]), 1))
                      .sum(1).mean())
        if v < best_val:
            best_val = v
            best_state = {k: t.detach().clone()
                          for k, t in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        E2 = torch.tensor(feats_eval, dtype=torch.float32, device=device)
        return model(P, E2).cpu().numpy()


def neighbour_scores(center: np.ndarray) -> np.ndarray:
    """Rank beams by angular distance to a centre beam (no learning)."""
    idx = np.arange(n_beams)
    return -np.abs(idx[None, :] - center[:, None]).astype(float)


def loss_curve(scores: np.ndarray, measured: np.ndarray | None) -> list:
    """Mean test power loss [dB] vs. total beams measured (1..MAX_BUDGET).
    measured: Set B indices (already measured) or None."""
    g = gains[te]
    order = np.argsort(-scores[te], axis=1)
    n_b = 0 if measured is None else len(measured)
    base = g[:, measured].max(1) if measured is not None else None
    curve = []
    for budget in range(1, MAX_BUDGET + 1):
        k = budget - n_b
        if k < 0:
            curve.append(None)
            continue
        if k == 0:
            picked = base
        else:
            extra = order[:, :k]
            picked = np.take_along_axis(g, extra, 1).max(1)
            if base is not None:
                picked = np.maximum(picked, base)
        assert picked is not None
        curve.append(float((g.max(1) - picked).mean()))
    return curve


def top1_acc(scores: np.ndarray) -> float:
    return float((scores[te].argmax(1) == best[te]).mean())


# ------------------------------------------------------------ experiments
t0 = time.time()
results: dict = {"n_test": int(te.sum()), "noise_db": NOISE_DB, "curves": {},
                 "top1": {}}

results["curves"]["geometric"] = loss_curve(neighbour_scores(geo), None)
results["top1"]["geometric"] = top1_acc(neighbour_scores(geo))

empty = np.zeros((N, 0), dtype=np.float32)
s = train_net(True, empty, empty)
results["curves"]["position NN"] = loss_curve(s, None)
results["top1"]["position NN"] = top1_acc(s)
print(f"  position NN done ({time.time() - t0:.0f} s)")

for step in SET_B_STEPS:
    set_b = np.arange(step // 2, n_beams, step)          # uniform subset
    tag = f"|B|={len(set_b)}"
    m_train, m_test = measure(set_b), measure(set_b)     # independent noise
    f_train, f_test = meas_features(m_train), meas_features(m_test)

    strongest = set_b[m_test.argmax(1)]
    sc = neighbour_scores(strongest)
    results["curves"][f"Set B strongest {tag}"] = loss_curve(sc, set_b)
    results["top1"][f"Set B strongest {tag}"] = top1_acc(sc)

    s = train_net(False, f_train, f_test)
    results["curves"][f"Set B NN {tag}"] = loss_curve(s, set_b)
    results["top1"][f"Set B NN {tag}"] = top1_acc(s)

    s = train_net(True, f_train, f_test)
    results["curves"][f"position + Set B NN {tag}"] = loss_curve(s, set_b)
    results["top1"][f"position + Set B NN {tag}"] = top1_acc(s)
    print(f"  Set B {tag} done ({time.time() - t0:.0f} s)")

with open(uos.tagged("results/metrics_setB.json"), "w") as f:
    json.dump(results, f, indent=2)

# ------------------------------------------------------------ summary
print(f"\nTest results on unseen areas ({results['n_test']} positions, "
      f"{NOISE_DB:.0f} dB measurement noise)")
print(f"{'method':<28}{'top-1':>7}   loss [dB] at 5 / 8 / 12 beams measured")
for name, curve in results["curves"].items():
    vals = [curve[b - 1] for b in (5, 8, 12)]
    txt = "  ".join(" n/a " if v is None else f"{v:5.2f}" for v in vals)
    print(f"{name:<28}{results['top1'][name] * 100:6.1f}%   {txt}")

# ------------------------------------------------------------ figure
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6b66", "#e6e5e1"
SERIES = [("position + Set B NN |B|=4", "Position + 4 measured beams (NN)",
           "#2a78d6"),
          ("Set B NN |B|=4", "4 measured beams only (NN)", "#eb6834"),
          ("Set B strongest |B|=4", "Strongest of 4 measured + neighbours (no AI)",
           "#1baf7a"),
          ("position NN", "Position only (NN)", "#eda100"),
          ("geometric", "Geometric beam (no AI)", "#e87ba4")]

fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
budgets = np.arange(1, MAX_BUDGET + 1)
for key, label, color in SERIES:
    curve = results["curves"][key]
    n_b = 4 if "|B|=4" in key else 0
    # Set B curves start at |B|+1: the Set B beams are always measured first
    xs = [b for b, v in zip(budgets, curve) if v is not None and b > n_b]
    ys = [v for b, v in zip(budgets, curve) if v is not None and b > n_b]
    ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round",
            marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=2, label=label)
ax.axhline(0, color=MUTED, linewidth=1)
ax.text(1, 0.12, f"Exhaustive sweep ({n_beams} beams) = 0 dB", color=MUTED,
        fontsize=9)
ax.set_ylim(0, 7)
ax.set_xticks(budgets)
ax.set_xlabel("Total beams measured (overhead)", color=MUTED)
ax.set_ylabel("Mean power loss vs. best beam [dB]", color=MUTED)
ax.set_title(f"Beam prediction with a few measured beams (BM-Case 1, {uos.FREQ_GHZ:g} GHz)",
             color=INK, loc="left", fontsize=13)
ax.grid(axis="y", color=GRID, linewidth=1)
ax.tick_params(colors=MUTED)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper center",
          bbox_to_anchor=(0.5, -0.16), ncol=2)
ax.text(0.0, -0.42, "Curves with measured beams start at 5: the 4 Set B beams "
         "are always measured first. Values above 7 dB are off scale.",
         color=MUTED, fontsize=8, transform=ax.transAxes)
fig.savefig(uos.tagged("figures/power_loss_vs_overhead.png"), dpi=200,
            bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
print("\nSaved", uos.tagged("figures/power_loss_vs_overhead.png"))
