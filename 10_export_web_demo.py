"""Step 10 - Export model predictions for the interactive web demo.

Retrains the position-only network (step 4) and the position + Set B (|B|=4)
network (step 5) with the same split and hyper-parameters, predicts the beam
ranking at every campus cell, and writes a compact JSON file:
  web/demo_data_<tag>.json
Gains are stored as uint8 (0.5 dB steps from -160 dB), base64-encoded.

Run:  python 10_export_web_demo.py [--freq 4.7]
"""
import base64
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import uos_twin as uos

SEED, BLOCK, SPLIT = 0, 40.0, (0.70, 0.15, 0.15)
N_FREQ, HIDDEN, DROPOUT, WD, EPOCHS, BATCH, LR, TEMP_DB = 3, 256, 0.1, 1e-3, 150, 1024, 2e-3, 2.0
SET_B = np.arange(0, 32, 8)          # 4 uniformly spaced beams of Set A
NOISE_DB, FLOOR_DB, TOPK = 1.0, -140.0, 8
OUT = Path("web")
OUT.mkdir(exist_ok=True)
torch.set_num_threads(8)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

D = np.load(uos.tagged("data/beams_uos.npz"))
valid = D["covered"] & D["in_sector"]
pos = np.stack([D["x"][valid], D["y"][valid], D["z"][valid]], 1)
gains = D["gains_db"][:, valid].T
best, geo = D["best"][valid], D["geo"][valid]
N, n_beams = gains.shape

rng = np.random.default_rng(SEED)
bx = np.floor(pos[:, 0] / BLOCK).astype(int)
by = np.floor(pos[:, 1] / BLOCK).astype(int)
blocks = np.unique(np.stack([bx, by], 1), axis=0)
rng.shuffle(blocks)
n_tr, n_va = int(SPLIT[0] * len(blocks)), int(SPLIT[1] * len(blocks))
block_id = {tuple(b): (0 if i < n_tr else 1 if i < n_tr + n_va else 2)
            for i, b in enumerate(blocks)}
part = np.array([block_id[(a, b)] for a, b in zip(bx, by)])
tr, va, te = part == 0, part == 1, part == 2

lo, hi = pos[tr].min(0), pos[tr].max(0)
pos_n = 2 * (pos - lo) / (hi - lo) - 1
soft = torch.softmax(torch.tensor((gains - gains.max(1, keepdims=True)) / TEMP_DB,
                                  dtype=torch.float32), dim=1)
noise = np.random.default_rng(SEED + 1)
meas = np.maximum(gains[:, SET_B] + NOISE_DB * noise.standard_normal((N, len(SET_B))), FLOOR_DB)
meas_f = np.concatenate([(meas + 100) / 20, (meas - meas.max(1, keepdims=True)) / 10], 1)


class BeamNet(nn.Module):
    def __init__(self, n_extra: int):
        super().__init__()
        self.freqs: torch.Tensor
        self.register_buffer("freqs", (2.0 ** torch.arange(N_FREQ)) * torch.pi)
        self.net = nn.Sequential(
            nn.Linear(3 + 6 * N_FREQ + n_extra, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, n_beams))

    def forward(self, p, e):
        ang = p[..., None] * self.freqs
        return self.net(torch.cat([p, torch.sin(ang).flatten(1), torch.cos(ang).flatten(1), e], 1))


def train(extra: np.ndarray) -> np.ndarray:
    torch.manual_seed(SEED)
    m = BeamNet(extra.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    P = torch.tensor(pos_n, dtype=torch.float32, device=device)
    E = torch.tensor(extra, dtype=torch.float32, device=device)
    Y = soft.to(device)
    i_tr = torch.tensor(np.flatnonzero(tr), device=device)
    i_va = torch.tensor(np.flatnonzero(va), device=device)
    best_v, state = float("inf"), None
    for _ in range(EPOCHS):
        m.train()
        perm = i_tr[torch.randperm(len(i_tr), device=device)]
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss = -(Y[b] * F.log_softmax(m(P[b], E[b]), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            v = float(-(Y[i_va] * F.log_softmax(m(P[i_va], E[i_va]), 1)).sum(1).mean())
        if v < best_v:
            best_v, state = v, {k: t.detach().clone() for k, t in m.state_dict().items()}
    assert state is not None
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        return m(P, E).cpu().numpy()


def report(name: str, order: np.ndarray, measured=None):
    g = gains[te]
    top1 = (order[te, 0] == best[te]).mean() * 100
    k = 4 if measured is None else 4
    picked = np.take_along_axis(g, order[te, :k], 1).max(1)
    if measured is not None:
        picked = np.maximum(picked, g[:, measured].max(1))
    print(f"{name:28s} top-1 {top1:5.1f} %   loss({'4' if measured is None else '4+4'} beams) "
          f"{(g.max(1) - picked).mean():.2f} dB")


print(f"{uos.FREQ_GHZ:g} GHz | samples {N} | test {te.sum()}")
order_pos = np.argsort(-train(np.zeros((N, 0))), 1)[:, :TOPK]
report("position NN", order_pos)
order_ps = np.argsort(-train(meas_f), 1)[:, :TOPK]
report("position + Set B(4) NN", order_ps, SET_B)

# ------------------------------------------------------------ export
H, W = D["x"].shape
q = np.clip(np.round((gains + 160) * 2), 0, 255).astype(np.uint8)       # (N, 32)
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
cell_index = np.flatnonzero(valid.ravel()).astype(np.uint32)

campus = uos.campus_polygon()
foot = uos.building_footprints()
rp = uos.tagged("data/route_gate_to_it.npz")
route = np.load(rp) if rp.exists() else None
data = {
    "freq_ghz": uos.FREQ_GHZ, "grid": [int(H), int(W)], "cell_m": float(D["cell"]),
    "x0": float(D["x"][0, 0]), "y0": float(D["y"][0, 0]),
    "angles": D["angles"].round(2).tolist(), "tx": D["tx_pos"].round(1).tolist(),
    "set_b": SET_B.tolist(), "n": int(N),
    "cell_index": b64(cell_index), "gains_q": b64(q), "best": b64(best.astype(np.uint8)),
    "geo": b64(geo.astype(np.uint8)), "part": b64(part.astype(np.uint8)),
    "order_pos": b64(order_pos.astype(np.uint8)), "order_ps": b64(order_ps.astype(np.uint8)),
    "meas_q": b64(np.clip(np.round((meas + 160) * 2), 0, 255).astype(np.uint8)),
    "campus": np.asarray(campus.exterior.coords).round(1).tolist(),
    "buildings": [np.asarray(f)[:, :2].round(1).tolist() for f in foot],
    "route": None if route is None else {"x": route["x"].round(1).tolist(), "y": route["y"].round(1).tolist(),
              "steps": route["steps"].tolist(), "ai": route["ai_beam"].tolist(),
              "best": route["best_beam"].tolist(), "p_best": route["p_best"].round(2).tolist(),
              "p_ai": route["p_ai"].round(2).tolist()},
}
out = OUT / f"demo_data_{uos.TAG}.json"
out.write_text(json.dumps(data, separators=(",", ":")))
print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
