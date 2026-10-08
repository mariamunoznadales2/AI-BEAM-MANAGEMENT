"""
Step 6 - A user walking across the campus: temporal beam prediction
(3GPP BM-Case 2) and an animated GIF.

Part A - Temporal beam prediction (BM-Case 2)
  The UE measures the 4 Set B beams at every step while walking (4 m per
  step, ~3 s at walking speed). A GRU network looks at the last 4 steps
  (positions + Set B measurements) and predicts the best of the 32 beams at
  the NEXT step, before the user gets there. Compared with:
    - full sweep, one step late : sweep all 32 beams, but the result is
                       used at the next step (the user has moved meanwhile)
    - Set B NN (now) : step-5 style network using only the current step
  Random walks for training stay inside training blocks and test walks
  inside test blocks (same spatial split as steps 4-5).
  Overhead: the full sweep measures 32 beams per step, the AI schemes
  4 (top-1) or 4 + 3 (top-3) beams per step.

Part B - Real campus routes (OSM walkways)
  1) the longest walkway route inside the campus -> 2D GIF and figure
  2) main gate -> Information and Technology Building -> saved for the 3D
     animation of step 7 (data/route_gate_to_it_<freq>.npz)
  At each step
  the UE measures the 4 Set B beams, the GRU predicts the top-3 beams for
  the next step and the strongest of those is used (7 of 32 beams measured).
  Every frame shows the true best beam, the AI-selected beam and the
  received power over time.

Run from ~/Desktop/uos-beam-twin with the environment active:
    python 06_walking_user.py              (3.5 GHz, default)
    python 06_walking_user.py --freq 4.7   (4.7 GHz)
Outputs (<freq> = 3p5GHz or 4p7GHz):
  results/metrics_temporal_<freq>.json
  data/route_gate_to_it_<freq>.npz
  figures/walking_user_<freq>.gif
  figures/walking_user_power_<freq>.png
"""
import heapq
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
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap    # noqa: E402
from matplotlib.patches import Polygon as MplPolygon     # noqa: E402
from scipy.spatial import KDTree                         # noqa: E402

import uos_twin as uos                                   # noqa: E402

SEED = 0
BLOCK = 40.0                  # same spatial split as steps 4-5
SPLIT = (0.70, 0.15, 0.15)
SET_B = np.arange(4, 32, 8)   # 4 measured beams (every 8th), as in step 5
NOISE_DB = 1.0
FLOOR_DB = -140.0
HIST = 4                      # past steps seen by the model
N_WALKS_TRAIN, N_WALKS_TEST = 3000, 600
WALK_LEN = 24                 # each walk gives WALK_LEN - HIST training windows
HIDDEN = 128
EPOCHS = 60
BATCH = 512
LR = 2e-3
TEMP_DB = 2.0
STEP_M = 4.0                  # walking step = one grid cell [m]
SECONDS_PER_STEP = 3.0        # ~1.4 m/s walking speed
FRAME_STRIDE = 2              # animation: one frame every 2 steps
ROUTE_START = (37.58350, 127.05499)   # UOS main gate (lat, lon)
ROUTE_END = (37.58298, 127.06084)     # Information and Technology Building
FIG_DIR, RES_DIR = Path("figures"), Path("results")
for p in (FIG_DIR, RES_DIR):
    p.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Training device:", device)

# ------------------------------------------------------------ data grid
D = np.load(uos.tagged("data/beams_uos.npz"))
print(f"Carrier frequency: {uos.FREQ_GHZ:g} GHz")
valid = D["covered"] & D["in_sector"]
ny, nx = valid.shape
G = D["gains_db"]                                   # (32, ny, nx)
angles = D["angles"]
n_beams = len(angles)
X, Y, Z = D["x"], D["y"], D["z"]

# Same spatial block split as steps 4-5 (identical seed and ordering)
pos_v = np.stack([X[valid], Y[valid], Z[valid]], 1)
bx = np.floor(pos_v[:, 0] / BLOCK).astype(int)
by = np.floor(pos_v[:, 1] / BLOCK).astype(int)
blocks = np.unique(np.stack([bx, by], 1), axis=0)
rng.shuffle(blocks)
n_tr = int(SPLIT[0] * len(blocks))
n_va = int(SPLIT[1] * len(blocks))
block_id = {tuple(b): (0 if i < n_tr else 1 if i < n_tr + n_va else 2)
            for i, b in enumerate(blocks)}
split = np.full((ny, nx), -1)
split[valid] = [block_id[(a, b)] for a, b in zip(bx, by)]
lo, hi = pos_v[split[valid] == 0].min(0), pos_v[split[valid] == 0].max(0)


def pos_norm(i, j):
    p = np.stack([X[i, j], Y[i, j], Z[i, j]], -1)
    return 2 * (p - lo) / (hi - lo) - 1


def measure(i, j):
    """Noisy Set B measurements [dB] at cells (i, j)."""
    m = np.moveaxis(G[SET_B][:, i, j], 0, -1)
    m = m + NOISE_DB * rng.standard_normal(m.shape)
    return np.maximum(m, FLOOR_DB)


def meas_features(m):
    return np.concatenate([(m + 100.0) / 20.0,
                           (m - m.max(-1, keepdims=True)) / 10.0], -1)


def soft_labels(i, j):
    g = np.moveaxis(G[:, i, j], 0, -1)
    return torch.softmax(torch.tensor((g - g.max(-1, keepdims=True)) / TEMP_DB,
                                      dtype=torch.float32), -1)


# ------------------------------------------------------------ random walks
MOVES = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def random_walks(part: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Walks of WALK_LEN cells with momentum, staying inside one split."""
    allowed = split == part
    cells = np.argwhere(allowed)
    I = np.zeros((n, WALK_LEN), dtype=int)
    J = np.zeros((n, WALK_LEN), dtype=int)
    for w in range(n):
        i, j = cells[rng.integers(len(cells))]
        d = MOVES[rng.integers(8)]
        for t in range(WALK_LEN):
            I[w, t], J[w, t] = i, j
            options = [d] if rng.random() < 0.8 else []
            options += [MOVES[k] for k in rng.permutation(8)]
            for di, dj in options:
                a, b = i + di, j + dj
                if 0 <= a < ny and 0 <= b < nx and allowed[a, b]:
                    i, j, d = a, b, (di, dj)
                    break
    return I, J


t0 = time.time()
I_tr, J_tr = random_walks(0, N_WALKS_TRAIN)
I_va, J_va = random_walks(1, N_WALKS_TEST // 2)
I_te, J_te = random_walks(2, N_WALKS_TEST)
print(f"Random walks: train {N_WALKS_TRAIN}, val {len(I_va)}, test {len(I_te)} "
      f"({time.time() - t0:.1f} s)")


def sequences(I, J):
    """Sliding windows: inputs = HIST steps [position, Set B features],
    target = best beam at the following step. Also returns the target cells
    and the cells of the last observed step."""
    feats = np.concatenate([pos_norm(I, J), meas_features(measure(I, J))], -1)
    xs, ti, tj, pi, pj = [], [], [], [], []
    for t in range(HIST, WALK_LEN):
        xs.append(feats[:, t - HIST:t])
        ti.append(I[:, t])
        tj.append(J[:, t])
        pi.append(I[:, t - 1])
        pj.append(J[:, t - 1])
    x = torch.tensor(np.concatenate(xs), dtype=torch.float32)
    ti, tj = np.concatenate(ti), np.concatenate(tj)
    pi, pj = np.concatenate(pi), np.concatenate(pj)
    return x, soft_labels(ti, tj), (ti, tj, pi, pj)


x_tr, y_tr, _ = sequences(I_tr, J_tr)
x_va, y_va, _ = sequences(I_va, J_va)
x_te, y_te, (TI, TJ, PI, PJ) = sequences(I_te, J_te)
print(f"Training windows: {len(x_tr)}  |  test windows: {len(x_te)}")


# ------------------------------------------------------------ models
class TemporalBeamNet(nn.Module):
    """GRU over the last HIST steps -> best beam at the next step."""

    def __init__(self, n_in: int):
        super().__init__()
        self.gru = nn.GRU(n_in, HIDDEN, num_layers=2, batch_first=True,
                          dropout=0.1)
        self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
                                  nn.Linear(HIDDEN, n_beams))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.head(h[:, -1])


class SnapshotBeamNet(nn.Module):
    """Same inputs but only the most recent step (no memory)."""

    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(),
                                 nn.Linear(256, n_beams))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, -1])


def train(model: nn.Module) -> nn.Module:
    torch.manual_seed(SEED)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    xt, yt = x_tr.to(device), y_tr.to(device)
    xv, yv = x_va.to(device), y_va.to(device)
    best_val, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for k in range(0, len(perm), BATCH):
            b = perm[k:k + BATCH]
            loss = -(yt[b] * F.log_softmax(model(xt[b]), 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = float(-(yv * F.log_softmax(model(xv), 1)).sum(1).mean())
        if v < best_val:
            best_val = v
            best_state = {k: t.detach().clone()
                          for k, t in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return model.eval()


def scores(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return model(x.to(device)).cpu().numpy()


t0 = time.time()
n_in = x_tr.shape[-1]
gru = train(TemporalBeamNet(n_in))
snap = train(SnapshotBeamNet(n_in))
print(f"Models trained in {time.time() - t0:.1f} s")

# ------------------------------------------------------------ evaluation
g_next = np.moveaxis(G[:, TI, TJ], 0, -1)                     # (N, 32)
g_prev = np.moveaxis(G[:, PI, PJ], 0, -1)
hold = g_prev.argmax(1)
idx = np.arange(n_beams)
methods = {
    "full sweep, one step late": -np.abs(idx[None] - hold[:, None]).astype(float),
    "Set B NN (current step only)": scores(snap, x_te),
    "GRU (last 4 steps)": scores(gru, x_te),
}


def evaluate(sc: np.ndarray) -> dict:
    order = np.argsort(-sc, 1)
    best_next = g_next.argmax(1)
    out = {"top1_acc": float((order[:, 0] == best_next).mean())}
    for k in (1, 3):
        picked = np.take_along_axis(g_next, order[:, :k], 1).max(1)
        out[f"loss_top{k}_db"] = float((g_next.max(1) - picked).mean())
    return out


results = {name: evaluate(sc) for name, sc in methods.items()}
results["setup"] = {"history_steps": HIST, "step_m": STEP_M,
                    "set_b": SET_B.tolist(), "noise_db": NOISE_DB,
                    "n_test_walks": int(len(I_te))}
with open(uos.tagged("results/metrics_temporal.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nPredicting the best beam one step ahead (test walks, unseen areas)")
print(f"{'method':<32}{'top-1':>7}{'loss top-1':>12}{'loss top-3':>12}")
for name in methods:
    r = results[name]
    print(f"{name:<32}{r['top1_acc'] * 100:6.1f}%{r['loss_top1_db']:9.2f} dB"
          f"{r['loss_top3_db']:9.2f} dB")

# ------------------------------------------------------------ campus route
campus = uos.campus_polygon()
xy, edges = uos.walkway_graph(campus)


def dijkstra(src: str):
    dist, prev = {src: 0.0}, {}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in edges.get(u, []):
            if d + w < dist.get(v, float("inf")):
                dist[v], prev[v] = d + w, u
                heapq.heappush(heap, (d + w, v))
    return dist, prev


def path_nodes(a: str, b: str) -> list[str]:
    dist, prev = dijkstra(a)
    assert b in dist, "The two route points are not connected by walkways"
    nodes = [b]
    while nodes[-1] != a:
        nodes.append(prev[nodes[-1]])
    return nodes[::-1]


def nearest_node(latlon: tuple[float, float]) -> str:
    p = np.array(uos.PROJ.project(*latlon))
    ids = list(xy)
    pts = np.array([xy[n] for n in ids])
    return ids[int(np.argmin(np.hypot(*(pts - p).T)))]


def run_route(nodes: list[str], label: str) -> dict:
    """Resample a walkway route every STEP_M metres, snap it to grid cells
    with signal and run the AI scheme along it. The AI scheme: measure the
    4 Set B beams, predict with the GRU, measure the top-3 predicted beams
    and use the strongest (4 + 3 = 7 of 32 beams per step)."""
    pts = np.array([xy[n] for n in nodes])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    s_new = np.arange(0, s[-1], STEP_M)
    route = np.stack([np.interp(s_new, s, pts[:, 0]),
                      np.interp(s_new, s, pts[:, 1])], 1)
    vi, vj = np.nonzero(valid)
    tree = KDTree(np.stack([X[vi, vj], Y[vi, vj]], 1))
    dist, nearest = tree.query(route)
    dist, nearest = np.asarray(dist), np.asarray(nearest, dtype=int)
    keep = dist < 6.0
    ri, rj = vi[nearest[keep]], vj[nearest[keep]]
    n = len(ri)
    feats = np.concatenate([pos_norm(ri, rj), meas_features(measure(ri, rj))], -1)
    windows = np.stack([feats[t - HIST:t] for t in range(HIST, n)])
    top3 = np.argsort(-scores(gru, torch.tensor(windows, dtype=torch.float32)),
                      1)[:, :3]
    g = np.moveaxis(G[:, ri, rj], 0, -1)                    # (n, 32)
    st = np.arange(HIST, n)
    chosen = top3[np.arange(len(st)), np.take_along_axis(g[st], top3, 1).argmax(1)]
    r = {"RI": ri, "RJ": rj, "steps": st, "pred": chosen, "g": g,
         "p_best": g[st].max(1), "p_ai": g[st, chosen],
         "p_hold": g[st, g[st - 1].argmax(1)], "t_sec": st * SECONDS_PER_STEP,
         "length_m": float(s[-1])}
    print(f"\n{label}: {r['length_m']:.0f} m along OSM walkways, "
          f"{n} of {len(route)} steps with signal")
    print(f"  AI (7 beams/step) loss {np.mean(r['p_best'] - r['p_ai']):.2f} dB"
          f" | full sweep one step late (32 beams/step) loss "
          f"{np.mean(r['p_best'] - r['p_hold']):.2f} dB")
    return r


# Route 1 (GIF): the longest walkway route inside the campus (graph diameter)
d0, _ = dijkstra(next(iter(edges)))
far_a = max(d0, key=lambda n: d0[n])
da, _ = dijkstra(far_a)
far_b = max(da, key=lambda n: da[n])
longest = run_route(path_nodes(far_a, far_b), "Longest campus route")

# Route 2 (3D animation, step 7): main gate -> IT Building (정보기술관)
gate = run_route(path_nodes(nearest_node(ROUTE_START), nearest_node(ROUTE_END)),
                 "Main gate -> IT Building")
Path("data").mkdir(exist_ok=True)
np.savez_compressed(
    uos.tagged("data/route_gate_to_it.npz"),
    x=X[gate["RI"], gate["RJ"]], y=Y[gate["RI"], gate["RJ"]],
    z=Z[gate["RI"], gate["RJ"]], steps=gate["steps"], ai_beam=gate["pred"],
    best_beam=gate["g"][gate["steps"]].argmax(1), p_best=gate["p_best"],
    p_ai=gate["p_ai"], t_sec=gate["t_sec"],
    start_xy=np.array(uos.PROJ.project(*ROUTE_START)),
    end_xy=np.array(uos.PROJ.project(*ROUTE_END)))
print("Saved", uos.tagged("data/route_gate_to_it.npz"), "(used by 07_walking_user_3d.py)")

# The 2D figures and the GIF use the longest route
RI, RJ, steps, pred = longest["RI"], longest["RJ"], longest["steps"], longest["pred"]
g_route, t_sec = longest["g"], longest["t_sec"]
p_best, p_ai, p_hold = longest["p_best"], longest["p_ai"], longest["p_hold"]

# ------------------------------------------------------------ figures
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6b66", "#e6e5e1"
C_BEST, C_AI, C_HOLD = "#6b6b66", "#2a78d6", "#eb6834"
beam_cmap = LinearSegmentedColormap.from_list(
    "beams", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9e1b1b"])
tx = D["tx_pos"]
R = uos.rotation_matrix_np(*D["tx_orientation"])
footprints = uos.building_footprints()
half = float(D["cell"]) / 2
extent = (X[0, 0] - half, X[0, -1] + half, Y[0, 0] - half, Y[-1, 0] + half)
best_map = np.where(valid, angles[G.argmax(0)], np.nan)


def beam_dir(k: int) -> np.ndarray:
    ph = np.radians(angles[k])
    v = R @ np.array([np.cos(ph), np.sin(ph), 0.0])
    return v[:2] / np.linalg.norm(v[:2])


def power_axes(ax):
    """Power loss vs. the best beam over time (0 dB = exhaustive sweep)."""
    ax.set_facecolor(SURFACE)
    ax.plot(t_sec, np.minimum(p_best - p_hold, 30), color=C_HOLD, linewidth=1.6,
            label="Full sweep, one step late (32 beams/step)")
    ax.plot(t_sec, np.minimum(p_best - p_ai, 30), color=C_AI, linewidth=1.6,
            label="AI: 4 + 3 beams/step")
    ax.set_ylim(31, -1)                      # smaller loss = higher
    ax.set_xlabel("Time [s]", color=MUTED)
    ax.set_ylabel("Loss vs. best [dB]", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=2)


# Static power-vs-time figure (for the report)
fig, ax = plt.subplots(figsize=(9, 3.4), facecolor=SURFACE)
power_axes(ax)
ax.set_title("Beam loss while walking across the campus "
             f"(mean: AI {np.mean(p_best - p_ai):.1f} dB, late sweep "
             f"{np.mean(p_best - p_hold):.1f} dB)", color=INK, loc="left",
             fontsize=11, pad=28)
fig.savefig(uos.tagged("figures/walking_user_power.png"), dpi=200, bbox_inches="tight",
            facecolor=SURFACE)
plt.close(fig)
print("Saved", uos.tagged("figures/walking_user_power.png"))

# Animated GIF
x_all, y_all = X[RI, RJ], Y[RI, RJ]
pad = 120
xlim = (min(x_all.min(), tx[0]) - pad, max(x_all.max(), tx[0]) + pad)
ylim = (min(y_all.min(), tx[1]) - pad, max(y_all.max(), tx[1]) + pad)
fig_w = 8.0
map_h = fig_w * 0.94 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
pow_h, gap, top = 1.6, 1.0, 0.5
fig_h = map_h + pow_h + gap + top
fig = plt.figure(figsize=(fig_w, fig_h), facecolor=SURFACE)
ax_map = fig.add_axes((0.03, (pow_h + gap) / fig_h, 0.94, map_h / fig_h))
ax_pow = fig.add_axes((0.10, 0.55 / fig_h, 0.86, (pow_h - 0.55) / fig_h))
ax_map.set_facecolor(SURFACE)
ax_map.imshow(best_map, origin="lower", extent=extent, cmap=beam_cmap,
              vmin=angles[0], vmax=angles[-1], alpha=0.35,
              interpolation="nearest")
for fp in footprints:
    ax_map.add_patch(MplPolygon(fp, closed=True, facecolor="#dcdad4",
                                edgecolor="#b9b7b0", linewidth=0.4))
cx, cy = campus.exterior.xy
ax_map.plot(cx, cy, color=INK, linewidth=1, linestyle=(0, (4, 3)))
ax_map.scatter([tx[0]], [tx[1]], marker="*", s=240, color=INK,
               edgecolor=SURFACE, linewidth=1.5, zorder=6)
ax_map.set_xlim(*xlim)
ax_map.set_ylim(*ylim)
ax_map.set_aspect("equal")
ax_map.axis("off")
title = ax_map.set_title("", color=INK, loc="left", fontsize=12)
trail, = ax_map.plot([], [], color=INK, linewidth=1.5, alpha=0.6)
true_ray, = ax_map.plot([], [], color=INK, linewidth=1.2,
                        linestyle=(0, (3, 2)), label="True best beam")
ai_ray, = ax_map.plot([], [], color=C_AI, linewidth=3, alpha=0.85,
                      label="AI-selected beam (7 of 32 measured)")
user = ax_map.scatter([], [], s=90, color=C_AI, edgecolor=SURFACE,
                      linewidth=2, zorder=7)
ax_map.legend(handles=[ai_ray, true_ray], frameon=False, fontsize=9,
              labelcolor=INK, loc="lower left")
power_axes(ax_pow)
cursor = ax_pow.axvline(t_sec[0], color=INK, linewidth=1)

frames = list(range(0, len(steps), FRAME_STRIDE))


def draw(f: int):
    k = steps[f]
    ux, uy = X[RI[k], RJ[k]], Y[RI[k], RJ[k]]
    r = np.hypot(ux - tx[0], uy - tx[1]) + 60
    for line, beam in ((true_ray, int(g_route[k].argmax())), (ai_ray, int(pred[f]))):
        dvec = beam_dir(beam)
        line.set_data([tx[0], tx[0] + r * dvec[0]], [tx[1], tx[1] + r * dvec[1]])
    trail.set_data(X[RI[:k + 1], RJ[:k + 1]], Y[RI[:k + 1], RJ[:k + 1]])
    user.set_offsets([[ux, uy]])
    cursor.set_xdata([t_sec[f], t_sec[f]])
    loss = p_best[f] - p_ai[f]
    title.set_text(f"Walking across the UOS campus  |  t = {t_sec[f]:4.0f} s"
                   f"  |  AI beam loss {loss:4.1f} dB")
    return trail, true_ray, ai_ray, user, cursor, title


anim = FuncAnimation(fig, draw, frames=frames, blit=False)
anim.save(str(uos.tagged("figures/walking_user.gif")), writer=PillowWriter(fps=8),
          dpi=90, savefig_kwargs={"facecolor": SURFACE})
plt.close(fig)
print(f"Saved {uos.tagged('figures/walking_user.gif')} ({len(frames)} frames)")
