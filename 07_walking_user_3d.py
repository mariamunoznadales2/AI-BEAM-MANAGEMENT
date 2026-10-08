"""
Step 7 - Cinematic 3D video: María walks from the UOS main gate to the
Information and Technology Building while the AI chooses the beam.

Dark "tech" look, rendered with Sionna RT:
  - dark campus model (buildings, roads, vegetation, terrain),
  - glowing ground = coverage of the beam selected by the AI at that moment
    (from the ray-traced beam dataset of step 3),
  - glowing lines = the strongest radio paths from the gNB to María,
    recomputed at every frame (white: line of sight, pink: reflection,
    amber: diffraction),
  - a blocky character (6x life size so it is visible) walking with arms and
    legs,
  - camera: establishing orbit over the campus -> fly-in -> chase camera,
  - HUD: title, live stats and a mini-map with the route.

Uses data from step 6 (data/route_gate_to_it_<freq>.npz).

Run from ~/Desktop/uos-beam-twin with the environment active:
    python 07_walking_user_3d.py --preview     quick low-quality test (~2 min)
    python 07_walking_user_3d.py               full quality (~10-15 min)
    python 07_walking_user_3d.py --freq 4.7    same, at 4.7 GHz
For the MP4 you need:  uv pip install imageio imageio-ffmpeg
Frames already rendered are reused, so an interrupted run continues where it
stopped. Delete the frames folder to start over.
Outputs:
  figures/walking_user_3d_<freq>.mp4   (or _preview.mp4)
  figures/walking_user_3d_<freq>.gif   small GIF preview
  figures/walking_user_3d_frames_<freq>[_preview]/frame_####.png
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, NoReturn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import mitsuba as mi                                     # noqa: E402
import numpy as np                                       # noqa: E402
from matplotlib import font_manager                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap    # noqa: E402
from matplotlib.patches import Polygon as MplPolygon     # noqa: E402
from PIL import Image, ImageDraw, ImageFont              # noqa: E402
from scipy.ndimage import gaussian_filter                # noqa: E402
import sionna.rt.renderer as sionna_renderer             # noqa: E402
import sionna.rt.utils.render as sionna_render           # noqa: E402
from sionna.rt import (Camera, PathSolver, PlanarArray,  # noqa: E402
                       RadioMapSolver, RadioMaterial, Receiver, SceneObject,
                       Transmitter)
from sionna.rt.constants import (INTERACTION_TYPE_TO_COLOR,  # noqa: E402
                                 InteractionType)

import uos_twin as uos                                   # noqa: E402

warnings.filterwarnings("ignore", message="The AST-transforming decorator")

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--preview", action="store_true")
PREVIEW: bool = _p.parse_known_args()[0].preview

# ------------------------------------------------------------ settings
NAME = "María"
if PREVIEW:
    RESOLUTION, NUM_SAMPLES, SUBSTEPS, N_INTRO, N_FLY, N_OUTRO = (
        (640, 360), 16, 1, 10, 8, 6)
else:
    RESOLUTION, NUM_SAMPLES, SUBSTEPS, N_INTRO, N_FLY, N_OUTRO = (
        (1280, 720), 64, 2, 36, 24, 24)
FPS = 12
FOV = 55.0
CHAR_SCALE = 6.0
LIGHTING = 1.1
RM_VMIN, RM_VMAX = -115.0, -70.0
N_ANT = 16
suffix = uos.TAG + ("_preview" if PREVIEW else "")
FIG_DIR = Path("figures")
FRAME_DIR = FIG_DIR / f"walking_user_3d_frames_{suffix}"
FRAME_DIR.mkdir(parents=True, exist_ok=True)
HIDDEN = np.array([0.0, 0.0, -5000.0])

# Dark palette (RGB 0-1)
MATERIAL_COLORS = {
    "wall": (0.17, 0.18, 0.22), "roof": (0.24, 0.25, 0.30),
    "road": (0.07, 0.075, 0.09), "vegetation": (0.05, 0.13, 0.11),
    "forest": (0.04, 0.11, 0.09), "park": (0.05, 0.13, 0.11),
    "water": (0.03, 0.10, 0.20), "pedestrian": (0.11, 0.11, 0.13),
    "footway": (0.11, 0.11, 0.13), "ground": (0.04, 0.045, 0.055),
}
RAY_COLORS = {"los": (1.0, 1.0, 1.0), "reflection": (1.0, 0.25, 0.70),
              "diffraction": (1.0, 0.75, 0.15)}
MAX_RAYS = 6          # only the strongest radio paths are drawn
COVERAGE_CMAP = LinearSegmentedColormap.from_list(
    "glow", ["#240046", "#5a189a", "#3a86ff", "#4cc9f0", "#e0fbfc"])

# Neon colours for the radio paths
sionna_render.LOS_COLOR = RAY_COLORS["los"]
INTERACTION_TYPE_TO_COLOR[InteractionType.SPECULAR] = RAY_COLORS["reflection"]
INTERACTION_TYPE_TO_COLOR[InteractionType.DIFFRACTION] = RAY_COLORS["diffraction"]
INTERACTION_TYPE_TO_COLOR[None] = RAY_COLORS["los"]


def strongest_path_segments(paths):
    """Like Sionna's paths_to_segments, but only for the MAX_RAYS strongest
    paths (synthetic array), so the picture stays readable."""
    vertices = paths.vertices.numpy()          # [depth, rx, tx, paths, 3]
    types = paths.interactions.numpy()         # [depth, rx, tx, paths]
    valid = paths.valid.numpy()                # [rx, tx, paths]
    if vertices.shape[-2] == 0:
        return [], [], []
    a_re, a_im = (np.asarray(t) for t in paths.a)   # [rx, rxa, tx, txa, paths]
    power = (a_re ** 2 + a_im ** 2).sum(axis=(1, 3))  # [rx, tx, paths]
    power = np.where(valid, power, -1.0)
    src = paths.sources.numpy().T
    tgt = paths.targets.numpy().T
    starts, ends, colors = [], [], []
    def ok(v):
        return bool(np.all(np.isfinite(v)) and np.all(np.abs(v) < 5000))

    for p in np.argsort(-power[0, 0])[:MAX_RAYS]:
        if power[0, 0, p] <= 0:
            continue
        seg_s, seg_e, seg_c = [], [], []
        start, color = src[0], sionna_render.LOS_COLOR
        good = True
        for i in range(vertices.shape[0]):
            t = types[i, 0, 0, p]
            if t == InteractionType.NONE:
                break
            if t not in INTERACTION_TYPE_TO_COLOR or not ok(vertices[i, 0, 0, p]):
                good = False           # corrupted entry: skip this path
                break
            seg_s.append(start)
            seg_e.append(vertices[i, 0, 0, p])
            seg_c.append(color)
            start, color = vertices[i, 0, 0, p], INTERACTION_TYPE_TO_COLOR[t]
        if not good:
            continue
        starts += seg_s + [start]
        ends += seg_e + [tgt[0]]
        colors += seg_c + [color]
    return starts, ends, colors


sionna_renderer.paths_to_segments = strongest_path_segments

# ------------------------------------------------------------ data
B = np.load(uos.tagged("data/beams_uos.npz"))
Rt = np.load(uos.tagged("data/route_gate_to_it.npz"))
angles = B["angles"]
G_db = B["gains_db"]                                       # (32, ny, nx)
covered = B["covered"]
tx_pos = B["tx_pos"].astype(float)
alpha, beta, gamma = (float(v) for v in B["tx_orientation"])
X, Y = B["x"], B["y"]
cell = float(B["cell"])
ux, uy, uz = Rt["x"], Rt["y"], Rt["z"] - uos.UE_HEIGHT     # ground under user
steps, ai_beam = Rt["steps"], Rt["ai_beam"]
loss_db, p_ai, t_sec = Rt["p_best"] - Rt["p_ai"], Rt["p_ai"], Rt["t_sec"]
print(f"Carrier {uos.FREQ_GHZ:g} GHz | route {len(ux)} points, "
      f"{len(steps)} steps | {'PREVIEW' if PREVIEW else 'full quality'}")

# ------------------------------------------------------------ scene
scene = uos.load_uos_scene()
for name, mat in scene.radio_materials.items():
    for key, col in MATERIAL_COLORS.items():
        if key in name.lower():
            mat.color = col
            break
    else:
        mat.color = MATERIAL_COLORS["ground"]

scene.tx_array = PlanarArray(num_rows=1, num_cols=N_ANT, vertical_spacing=0.5,
                             horizontal_spacing=0.5, pattern="tr38901",
                             polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                             polarization="V")
tx = Transmitter(name="gNB", position=mi.Point3f(*map(float, tx_pos)),
                 orientation=mi.Point3f(alpha, beta, gamma), display_radius=7,
                 color=(1.0, 0.2, 0.2))
rx = Receiver(name="ue", position=mi.Point3f(*map(float, HIDDEN)),
              display_radius=0.5, color=(1.0, 0.55, 0.1))
scene.add(tx)
scene.add(rx)


# ------------------------------------------------------------ meshes
def box(c, size):
    c, h = np.asarray(c, float), np.asarray(size, float) / 2
    v = np.array([[sx, sy, sz] for sz in (-1, 1) for sy in (-1, 1)
                  for sx in (-1, 1)], float) * h + c
    f = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],
                  [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
                  [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
    return v, f


def rot_y(v, pivot, ang):
    c, s = np.cos(ang), np.sin(ang)
    M = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (v - pivot) @ M.T + pivot


def character(phase: float):
    """Blocky character (feet at origin, facing +x) at a walking phase."""
    swing = 0.55 * np.sin(phase)
    parts = {"shirt": [], "legs": [], "head": [], "hair": []}
    for side in (-1, 1):
        v, f = box((0, 0.12 * side, 0.45), (0.2, 0.18, 0.9))
        parts["legs"].append((rot_y(v, np.array([0, 0, 0.9]), side * swing), f))
    parts["shirt"].append(box((0, 0, 1.22), (0.26, 0.46, 0.66)))
    for side in (-1, 1):
        v, f = box((0, 0.31 * side, 1.2), (0.12, 0.12, 0.62))
        parts["shirt"].append((rot_y(v, np.array([0, 0, 1.5]), -side * swing), f))
    parts["head"].append(box((0, 0, 1.72), (0.3, 0.3, 0.3)))
    parts["hair"].append(box((-0.03, 0, 1.84), (0.34, 0.34, 0.1)))   # hair
    parts["hair"].append(box((-0.15, 0, 1.6), (0.06, 0.32, 0.38)))   # long hair
    out = {}
    for name, items in parts.items():
        vs, fs, off = [], [], 0
        for v, f in items:
            vs.append(v)
            fs.append(f + off)
            off += len(v)
        out[name] = (np.concatenate(vs), np.concatenate(fs))
    return out


def make_object(name, verts, faces, color):
    mesh = mi.Mesh(name, vertex_count=len(verts), face_count=len(faces),
                   has_vertex_normals=False, has_vertex_texcoords=False)
    params = mi.traverse(mesh)
    params["vertex_positions"] = mi.Float(verts.astype(np.float32).ravel())
    params["faces"] = mi.UInt32(faces.astype(np.uint32).ravel())
    params.update()
    mat = RadioMaterial(name=f"{name}-mat", relative_permittivity=1.0,
                        conductivity=0.0, color=color)
    return SceneObject(mi_mesh=mesh, name=name, radio_material=mat)


def set_vertices(obj: SceneObject, verts: np.ndarray):
    params = scene.mi_scene_params
    params[obj.mi_mesh.id() + ".vertex_positions"] = mi.Float(
        verts.astype(np.float32).ravel())
    params.update()
    scene.scene_geometry_updated()


CHAR_COLORS = {"shirt": (1.0, 0.55, 0.12), "legs": (0.85, 0.87, 0.92),
               "head": (0.96, 0.80, 0.66), "hair": (0.20, 0.12, 0.08)}
pose0 = character(0.0)
objs = {k: make_object(f"walker-{k}", pose0[k][0] + HIDDEN, pose0[k][1], c)
        for k, c in CHAR_COLORS.items()}
scene.edit(add=list(objs.values()))


def place_character(p, heading, phase):
    c, s = np.cos(heading), np.sin(heading)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for name, (v, _) in character(phase).items():
        set_vertices(objs[name], (v * CHAR_SCALE) @ Rz.T + p)


def hide_character():
    for name, (v, _) in character(0.0).items():
        set_vertices(objs[name], v + HIDDEN)


# ------------------------------------- glowing coverage of the AI beam
# Grid of the step-3 dataset, raised 0.4 m above the 1.5 m handset height
xs = np.concatenate([X[0] - cell / 2, [X[0, -1] + cell / 2]])
ys = np.concatenate([Y[:, 0] - cell / 2, [Y[-1, 0] + cell / 2]])
Zv = np.array([[uos.ground_z(scene, x, y) for x in xs] for y in ys])
surface = uos.grid_to_mesh(xs, ys, Zv + uos.UE_HEIGHT + 0.4)
rm_template = RadioMapSolver()(scene=scene, measurement_surface=surface,
                               samples_per_tx=1000)
lin = 10 ** (G_db / 10) * covered[None]
mask_blur = gaussian_filter(covered.astype(float), 1.2) + 1e-9
glow_cache: dict[int, mi.TensorXf] = {}


def glow_tensor(beam: int) -> mi.TensorXf:
    """Smoothed coverage of one beam; weak cells transparent (0)."""
    if beam not in glow_cache:
        g = gaussian_filter(lin[beam], 1.2) / mask_blur
        g = np.where(covered & (10 * np.log10(g + 1e-30) > RM_VMIN), g, 0.0)
        per_tri = np.concatenate([g.ravel(), g.ravel()]).astype(np.float32)
        glow_cache[beam] = mi.TensorXf(per_tri[None, :])
    return glow_cache[beam]


current_glow = {"t": glow_tensor(int(ai_beam[0]))}


class _GlowMap(type(rm_template)):
    @property
    def path_gain(self):
        return current_glow["t"]


glow_map = rm_template
glow_map.__class__ = _GlowMap
paths_solver = PathSolver()

# ------------------------------------------------------------ timeline
c_x, c_y = float(np.mean(ux)), float(np.mean(uy))
c_z = uos.ground_z(scene, c_x, c_y)


def heading_at(t: float) -> float:
    a = int(np.clip(np.floor(t) - 3, 0, len(ux) - 1))
    b = int(np.clip(np.floor(t) + 3, 0, len(ux) - 1))
    return float(np.arctan2(uy[b] - uy[a], ux[b] - ux[a]))


def pos_at(t: float) -> np.ndarray:
    """Interpolated ground position along the route (t in route indices)."""
    i = int(np.clip(np.floor(t), 0, len(ux) - 2))
    w = float(np.clip(t - i, 0, 1))
    return np.array([ux[i] * (1 - w) + ux[i + 1] * w,
                     uy[i] * (1 - w) + uy[i + 1] * w,
                     uz[i] * (1 - w) + uz[i + 1] * w])


def chase_camera(t: float) -> tuple[np.ndarray, np.ndarray]:
    p, h = pos_at(t), heading_at(t)
    back = -np.array([np.cos(h), np.sin(h), 0.0])
    side = np.array([-np.sin(h), np.cos(h), 0.0])
    cam = p + 110 * back + 20 * side + np.array([0, 0, 110.0])
    look = p + 25 * (-back) + np.array([0, 0, 6.0])
    return cam, look


def orbit_camera(a: float) -> tuple[np.ndarray, np.ndarray]:
    ang = np.radians(200 + 60 * a)
    cam = np.array([c_x + 700 * np.cos(ang), c_y + 700 * np.sin(ang),
                    c_z + 430.0])
    return cam, np.array([c_x, c_y, c_z + 20])


def smooth(a: float) -> float:
    return a * a * (3 - 2 * a)


frames_plan = []      # (camera, look_at, route_t, step_index, walking)
for i in range(N_INTRO):
    cam, look = orbit_camera(i / max(N_INTRO - 1, 1))
    frames_plan.append((cam, look, 0.0, 0, False))
cam0, look0 = orbit_camera(1.0)
cam1, look1 = chase_camera(float(steps[0]))
for i in range(N_FLY):
    a = smooth((i + 1) / N_FLY)
    frames_plan.append((cam0 * (1 - a) + cam1 * a, look0 * (1 - a) + look1 * a,
                        float(steps[0]), 0, False))
cam_s = cam1
for f in range(len(steps)):
    for sub in range(SUBSTEPS):
        t = float(steps[f]) + sub / SUBSTEPS
        cam_t, look = chase_camera(t)
        cam_s = 0.8 * cam_s + 0.2 * cam_t
        frames_plan.append((cam_s.copy(), look, t, f, True))
t_end = float(steps[-1])
for i in range(N_OUTRO):
    a = smooth((i + 1) / N_OUTRO)
    cam_t, look = chase_camera(t_end)
    cam_o = cam_t + np.array([0, 0, 60 * a])
    frames_plan.append((cam_o, look, t_end, len(steps) - 1, False))
print(f"Frames to render: {len(frames_plan)} "
      f"({len(frames_plan) / FPS:.0f} s of video at {FPS} fps)")

# ------------------------------------------------------------ HUD
font_reg = font_manager.findfont("DejaVu Sans")
font_bold = font_manager.findfont(font_manager.FontProperties(
    family="DejaVu Sans", weight="bold"))
ui = RESOLUTION[0] / 1280
F_TITLE = ImageFont.truetype(font_bold, int(26 * ui))
F_SUB = ImageFont.truetype(font_reg, int(15 * ui))
F_LABEL = ImageFont.truetype(font_reg, int(12 * ui))
F_VALUE = ImageFont.truetype(font_bold, int(20 * ui))
WHITE, MUTED = (240, 242, 247), (150, 156, 170)

# Mini-map rendered once with matplotlib (dark)
campus = uos.campus_polygon()
footprints = uos.building_footprints()
mm_pad = 60
mm_x = (min(ux.min(), tx_pos[0]) - mm_pad, max(ux.max(), tx_pos[0]) + mm_pad)
mm_y = (min(uy.min(), tx_pos[1]) - mm_pad, max(uy.max(), tx_pos[1]) + mm_pad)
mm_w = int(300 * ui)
mm_h = int(mm_w * (mm_y[1] - mm_y[0]) / (mm_x[1] - mm_x[0]))
fig = plt.figure(figsize=(mm_w / 100, mm_h / 100), dpi=100)
ax = fig.add_axes((0, 0, 1, 1))
ax.set_facecolor("#0b0d12")
for fp in footprints:
    ax.add_patch(MplPolygon(fp, closed=True, facecolor="#2a2e38",
                            edgecolor="none"))
cx_, cy_ = campus.exterior.xy
ax.plot(cx_, cy_, color="#5a6070", linewidth=0.8, linestyle=(0, (3, 2)))
ax.plot(ux, uy, color="#4cc9f0", linewidth=1.6, alpha=0.9)
ax.scatter([tx_pos[0]], [tx_pos[1]], marker="*", s=90, color="#ff3b3b", zorder=5)
ax.set_xlim(*mm_x)
ax.set_ylim(*mm_y)
ax.axis("off")
mm_path = FRAME_DIR / "_minimap.png"
fig.savefig(mm_path, dpi=100, facecolor="#0b0d12")
plt.close(fig)
minimap = Image.open(mm_path).convert("RGBA").resize((mm_w, mm_h))


def to_minimap(x, y):
    return ((x - mm_x[0]) / (mm_x[1] - mm_x[0]) * mm_w,
            (1 - (y - mm_y[0]) / (mm_y[1] - mm_y[0])) * mm_h)


def panel(draw, box_, radius=12):
    draw.rounded_rectangle(box_, radius=int(radius * ui), fill=(10, 12, 18, 185))


def hud(png: Path, plan_i: int):
    cam, look, t, f, walking = frames_plan[plan_i]
    img = Image.open(png).convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    m = int(20 * ui)
    # Title
    panel(d, (m, m, m + int(640 * ui), m + int(78 * ui)))
    d.text((m + int(18 * ui), m + int(12 * ui)),
           f"{NAME.upper()}  ·  MAIN GATE → IT BUILDING", font=F_TITLE, fill=WHITE)
    d.text((m + int(18 * ui), m + int(48 * ui)),
           f"AI beam management in a digital twin of the University of Seoul  ·  "
           f"{uos.FREQ_GHZ:g} GHz", font=F_SUB, fill=MUTED)
    # Mini-map
    W, H = img.size
    mx, my = W - mm_w - m, m
    panel(d, (mx - 8, my - 8, mx + mm_w + 8, my + mm_h + 8))
    over.alpha_composite(minimap, (mx, my))
    p = pos_at(t)
    px, py = to_minimap(p[0], p[1])
    r = int(6 * ui)
    d.ellipse((mx + px - r, my + py - r, mx + px + r, my + py + r),
              fill=(255, 140, 30, 255), outline=(255, 255, 255, 255))
    # Stats
    beam = int(ai_beam[f])
    loss = float(loss_db[f])
    lc = (80, 220, 140) if loss < 1 else (255, 190, 60) if loss < 3 else (255, 90, 90)
    stats = [("TIME", f"{t_sec[f]:.0f} s" if walking else "—", WHITE),
             ("AI BEAM", f"{angles[beam]:+.1f}°", WHITE),
             ("BEAMS MEASURED", "7 / 32", WHITE),
             ("SIGNAL", f"{p_ai[f]:.0f} dB", WHITE),
             ("LOSS VS BEST", f"{loss:.1f} dB", lc)]
    bw, bh = int(150 * ui), int(62 * ui)
    y0 = H - m - bh
    for i, (lab, val, col) in enumerate(stats):
        x0 = m + i * (bw + int(10 * ui))
        panel(d, (x0, y0, x0 + bw, y0 + bh))
        d.text((x0 + int(14 * ui), y0 + int(8 * ui)), lab, font=F_LABEL, fill=MUTED)
        d.text((x0 + int(14 * ui), y0 + int(26 * ui)), val, font=F_VALUE, fill=col)
    # Legend
    items = [("Radio path: line of sight", RAY_COLORS["los"]),
             ("Radio path: reflection", RAY_COLORS["reflection"]),
             ("Radio path: diffraction", RAY_COLORS["diffraction"])]
    lw, lh = int(240 * ui), int(118 * ui)
    lx, ly = W - m - lw, H - m - lh
    panel(d, (lx, ly, lx + lw, ly + lh))
    for i, (lab, col) in enumerate(items):
        yy = ly + int(14 * ui) + i * int(20 * ui)
        c255 = tuple(int(255 * v) for v in col)
        d.line((lx + int(14 * ui), yy + int(7 * ui), lx + int(40 * ui),
                yy + int(7 * ui)), fill=c255, width=max(2, int(3 * ui)))
        d.text((lx + int(50 * ui), yy), lab, font=F_LABEL, fill=WHITE)
    # Coverage colour bar
    yy = ly + int(98 * ui)
    d.text((lx + int(14 * ui), yy - int(18 * ui)), "AI-beam coverage  (weak → strong)",
           font=F_LABEL, fill=MUTED)
    bar_w = lw - int(28 * ui)
    for k in range(bar_w):
        c = COVERAGE_CMAP(k / bar_w)
        d.line((lx + int(14 * ui) + k, yy, lx + int(14 * ui) + k, yy + int(8 * ui)),
               fill=tuple(int(255 * v) for v in c[:3]))
    Image.alpha_composite(img, over).convert("RGB").save(png)


# ------------------------------------------------------------ render loop
def render(png: Path, cam: np.ndarray, look: np.ndarray, paths) -> bool:
    """Render one frame. Retries with a nudged camera and, if needed,
    without radio paths. Returns False if every attempt failed."""
    for attempt in range(6):
        nudge = np.array([0.7, -0.5, 1.1]) * (attempt % 3)
        use_paths = paths if attempt < 3 else None
        camera = Camera(position=mi.Point3f(*map(float, cam + nudge)),
                        look_at=mi.Point3f(*map(float, look)))
        try:
            scene.render_to_file(camera=camera, filename=str(png),
                                 radio_map=glow_map, rm_vmin=RM_VMIN,
                                 rm_vmax=RM_VMAX, rm_cmap=COVERAGE_CMAP,
                                 paths=use_paths, resolution=RESOLUTION,
                                 num_samples=NUM_SAMPLES, fov=FOV,
                                 lighting_scale=LIGHTING)
        except (RuntimeError, KeyError, ValueError) as err:
            print(f"\n  render issue, retrying ({type(err).__name__})")
            continue
        if np.asarray(Image.open(png).convert("L")).mean() > 2.0:
            return True                # not a black frame
        print("\n  black frame, retrying")
    return False


def restart(reason: str) -> NoReturn:
    """The GPU (Metal) backend occasionally gets into a bad state after a
    failed render. Start a fresh Python process: rendered frames are kept,
    so it continues from the next frame."""
    print(f"\n  {reason} -> restarting the script to reset the GPU state...",
          flush=True)
    os.execv(sys.executable, [sys.executable] + sys.argv)


t0 = time.time()
rendered = 0
frame_files = []
for i, (cam, look, t, f, walking) in enumerate(frames_plan):
    png = FRAME_DIR / f"frame_{i:04d}.png"
    frame_files.append(png)
    if png.exists():
        continue
    p = pos_at(t)
    # 1) radio paths to the user (character out of the way)
    hide_character()
    rx.position = mi.Point3f(*map(float, p + np.array([0, 0, uos.UE_HEIGHT])))
    try:
        paths = paths_solver(scene=scene, max_depth=3, samples_per_src=10**6,
                             diffraction=True, refraction=False)
    except RuntimeError:
        restart(f"radio paths failed at frame {i}")
    # 2) glow of the AI-selected beam
    current_glow["t"] = glow_tensor(int(ai_beam[f]))
    # 3) character
    phase = t * np.pi if walking else 0.0
    place_character(p, heading_at(t), phase)
    # 4) render + HUD
    if not render(png, cam, look, paths):
        if i > 0 and frame_files[i - 1].exists():
            # Give up on this frame: repeat the previous one (HUD already on)
            Image.open(frame_files[i - 1]).save(png)
            print(f"\n  frame {i} could not be rendered, reusing the previous one")
            restart("render failed")
        raise RuntimeError(f"Could not render frame {i}")
    hud(png, i)
    rendered += 1
    left = len(frames_plan) - i - 1
    eta = (time.time() - t0) / rendered * left
    print(f"\r  frame {i + 1}/{len(frames_plan)}  (about {eta / 60:.1f} min left)",
          end="")
print(f"\nRendered {rendered} new frames in {(time.time() - t0) / 60:.1f} min")

# ------------------------------------------------------------ video + GIF
mp4 = FIG_DIR / f"walking_user_3d_{suffix}.mp4"
try:
    import imageio.v2 as imageio
    writer: Any = imageio.get_writer(mp4, fps=FPS, codec="libx264", quality=8,
                                     pixelformat="yuv420p", macro_block_size=8)
    for fp in frame_files:
        writer.append_data(np.asarray(Image.open(fp).convert("RGB")))
    writer.close()
    print(f"Saved {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)")
except ImportError:
    print("For the MP4 install:  uv pip install imageio imageio-ffmpeg")

gif = FIG_DIR / f"walking_user_3d_{suffix}.gif"
small = [Image.open(fp).convert("RGB").resize((480, 270))
         .convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
         for fp in frame_files[::2]]
small[0].save(gif, save_all=True, append_images=small[1:],
              duration=int(2000 / FPS), loop=0, optimize=True)
print(f"Saved {gif} ({gif.stat().st_size / 1e6:.1f} MB)")
