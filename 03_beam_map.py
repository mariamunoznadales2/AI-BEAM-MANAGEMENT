"""
Step 3 - Beam management: which beam is best at each point of the campus.

The gNB now has a 16-element horizontal array and a codebook of 32 beams
(steering directions from -60 to +60 degrees). For every beam we ray-trace
the received power over the whole campus -> this is the exhaustive beam
sweep a 5G network performs today. From it we get, for each position, the
best beam: the "ground-truth label" the neural network will learn next.

Run from ~/Desktop/uos-beam-twin with the environment active:
    python 03_beam_map.py              (3.5 GHz, default)
    python 03_beam_map.py --freq 4.7   (4.7 GHz)
Outputs (<freq> = 3p5GHz or 4p7GHz):
  data/beams_uos_<freq>.npz               dataset (positions, per-beam gain, best beam)
  figures/beam_map_<freq>.png             best beam at each location
  figures/geometric_beam_loss_<freq>.png  where "point straight at the user" fails
  figures/beam_map_3d_<freq>.png          3D view of the beam map
"""
import copy
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap    # noqa: E402
from matplotlib.patches import Polygon as MplPolygon     # noqa: E402

import mitsuba as mi                                     # noqa: E402
from sionna.rt import (Camera, PlanarArray, RadioMapSolver,  # noqa: E402
                       Transmitter)
import uos_twin as uos                                   # noqa: E402

N_ANT = 16             # array elements (horizontal, lambda/2 spacing)
N_BEAMS = 32           # codebook size
MAX_ANGLE = 60.0       # codebook covers -60..+60 degrees
CELL = 4.0             # cell size [m]
MARGIN = 120.0         # extent around the campus [m]
SAMPLES = 10**7        # rays per beam
MIN_GAIN_DB = -130.0   # below this: no coverage
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------ scene and gNB
scene = uos.load_uos_scene()
campus = uos.campus_polygon()
tx_x, tx_y, roof_z, _ = uos.highest_roof(scene, campus)
tx_pos = np.array([tx_x, tx_y, roof_z + uos.MAST])

scene.tx_array = PlanarArray(num_rows=1, num_cols=N_ANT,
                             vertical_spacing=0.5, horizontal_spacing=0.5,
                             pattern="tr38901", polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                             polarization="V")
tx = Transmitter(name="gNB", position=mi.Point3f(*map(float, tx_pos)),
                 display_radius=6)
scene.add(tx)
cen = campus.centroid
tx.look_at(mi.Point3f(cen.x, cen.y, uos.ground_z(scene, cen.x, cen.y)))
alpha, beta, gamma = (float(v[0]) for v in tx.orientation)
R = uos.rotation_matrix_np(alpha, beta, gamma)      # local array axes
print(f"Carrier frequency: {uos.FREQ_GHZ:g} GHz")
print(f"gNB at ({tx_x:.0f}, {tx_y:.0f}, {tx_pos[2]:.0f}) m, "
      f"boresight azimuth {np.degrees(alpha):.0f} deg")

# ------------------------------------------------------------ DFT codebook
# Precoder steering towards angle phi (in the array's local frame):
#   p_k = exp(-j 2 pi / lambda * r(phi) . d_k) / sqrt(N)
lam = float(scene.wavelength[0])
d = np.array(scene.tx_array.positions(lam))          # element positions
d = d if d.shape[0] == 3 else d.T                     # -> (3, N_ANT)
angles = np.linspace(-MAX_ANGLE, MAX_ANGLE, N_BEAMS)
codebook = []
for phi in np.radians(angles):
    r = np.array([np.cos(phi), np.sin(phi), 0.0])
    codebook.append(np.exp(-1j * 2 * np.pi / lam * (r @ d)) / np.sqrt(N_ANT))

# ------------------------------------------------------ measurement surface
xmin, ymin, xmax, ymax = campus.bounds
xs, ys, Z = uos.terrain_grid(scene, xmin - MARGIN, xmax + MARGIN,
                             ymin - MARGIN, ymax + MARGIN, CELL)
mesh = uos.grid_to_mesh(xs, ys, Z)
ny, nx = len(ys) - 1, len(xs) - 1
n_cells = nx * ny
print(f"Grid: {nx} x {ny} = {n_cells} cells of {CELL:.0f} m")

# --------------------------------------------- beam sweep (ray tracing)
solver = RadioMapSolver()
gains = np.zeros((N_BEAMS, ny, nx), dtype=np.float32)
t0 = time.time()
rm_last = None
for i, p in enumerate(codebook):
    rm = solver(scene=scene, measurement_surface=mesh,
                precoding_vec=(mi.TensorXf(p.real.astype(np.float32)),
                               mi.TensorXf(p.imag.astype(np.float32))),
                samples_per_tx=SAMPLES, max_depth=5, diffraction=True,
                seed=42)                   # same rays for every beam
    g = np.array(rm.path_gain).reshape(-1)
    gains[i] = (0.5 * (g[:n_cells] + g[n_cells:])).reshape(ny, nx)
    rm_last = rm
    print(f"\r  beam {i + 1:2d}/{N_BEAMS} ({angles[i]:+5.1f} deg)", end="")
print(f"\nBeam sweep finished in {time.time() - t0:.1f} s")

# ------------------------------------------------------------ labels
gains_db = 10 * np.log10(np.maximum(gains, 1e-20))
best = gains_db.argmax(0)
best_db = gains_db.max(0)
covered = best_db > MIN_GAIN_DB

# "Geometric" beam: the one pointing straight at the user
XC, YC = np.meshgrid(0.5 * (xs[:-1] + xs[1:]), 0.5 * (ys[:-1] + ys[1:]))
ZC = 0.25 * (Z[:-1, :-1] + Z[:-1, 1:] + Z[1:, :-1] + Z[1:, 1:])
u = np.stack([XC - tx_pos[0], YC - tx_pos[1], ZC - tx_pos[2]], -1) @ R
phi_user = np.degrees(np.arctan2(u[..., 1], u[..., 0]))
geo = np.abs(phi_user[..., None] - angles).argmin(-1)
geo_db = np.take_along_axis(gains_db, geo[None], 0)[0]
loss_db = best_db - geo_db

# Statistics only inside the antenna sector (|angle| <= 60 deg)
in_sector = np.abs(phi_user) <= MAX_ANGLE
ok = covered & in_sector
mismatch = (np.abs(best - geo) > 1) & ok
print(f"Covered cells: {covered.mean() * 100:.1f} % "
      f"(inside the sector: {ok.sum()} cells)")
print(f"Best beam is NOT the one pointing at the user (>1 beam off): "
      f"{mismatch.sum() / ok.sum() * 100:.1f} % of cells")
print(f"Mean loss of 'point straight at the user': {loss_db[ok].mean():.1f} dB"
      f" (90th percentile: {np.percentile(loss_db[ok], 90):.1f} dB)")

# ------------------------------------------------------------ dataset
Path("data").mkdir(exist_ok=True)
lat, lon = np.vectorize(uos.PROJ.unproject)(XC, YC)
np.savez_compressed(
    uos.tagged("data/beams_uos.npz"),
    x=XC.astype(np.float32), y=YC.astype(np.float32), z=ZC.astype(np.float32),
    lat=lat, lon=lon, gains_db=gains_db.astype(np.float32),
    best=best, covered=covered, in_sector=in_sector, geo=geo, angles=angles,
    tx_pos=tx_pos, tx_orientation=np.array([alpha, beta, gamma]), cell=CELL)
print("Saved", uos.tagged("data/beams_uos.npz"))

# ------------------------------------------------------------ 2D figures
SURFACE, INK, MUTED = "#fcfcfb", "#1f1f1e", "#6b6b66"
BUILD_FILL, BUILD_EDGE = "#dcdad4", "#b9b7b0"
beam_cmap = LinearSegmentedColormap.from_list(
    "beams", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9e1b1b"])
loss_cmap = LinearSegmentedColormap.from_list(
    "loss", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
extent = (xs[0], xs[-1], ys[0], ys[-1])
footprints = uos.building_footprints()
cx_, cy_ = campus.exterior.xy


def base_axes(title: str):
    fig, ax = plt.subplots(figsize=(9, 8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=10)
    ax.set_xlabel("x [m] (east)", color=MUTED)
    ax.set_ylabel("y [m] (north)", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax


def overlay(ax):
    for fp in footprints:
        ax.add_patch(MplPolygon(fp, closed=True, facecolor=BUILD_FILL,
                                edgecolor=BUILD_EDGE, linewidth=0.4, zorder=3))
    ax.plot(cx_, cy_, color=INK, linewidth=1.2, linestyle=(0, (4, 3)),
            zorder=4, label="UOS campus")
    ax.scatter([tx_pos[0]], [tx_pos[1]], marker="*", s=260, color=INK,
               edgecolor=SURFACE, linewidth=1.5, zorder=6, label="gNB")


def colorbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(label, color=MUTED)
    cb.ax.tick_params(colors=MUTED)
    cb.outline.set_visible(False)


# Best-beam map
fig, ax = base_axes(f"Best beam at each location (32 beams, {uos.FREQ_GHZ:g} GHz)")
img = np.where(covered, angles[best], np.nan)
im = ax.imshow(img, origin="lower", extent=extent, cmap=beam_cmap,
               vmin=-MAX_ANGLE, vmax=MAX_ANGLE, interpolation="nearest",
               zorder=2)
for k in range(0, N_BEAMS, 4):        # fan showing some beam directions
    ph = np.radians(angles[k])
    dvec = R @ np.array([np.cos(ph), np.sin(ph), 0.0])
    dvec = dvec[:2] / np.linalg.norm(dvec[:2])
    ax.plot([tx_pos[0], tx_pos[0] + 900 * dvec[0]],
            [tx_pos[1], tx_pos[1] + 900 * dvec[1]],
            color=beam_cmap((angles[k] + MAX_ANGLE) / (2 * MAX_ANGLE)),
            linewidth=0.8, alpha=0.9, zorder=5)
overlay(ax)
colorbar(fig, ax, im, "Best beam direction [deg]")
ax.legend(loc="lower left", frameon=False, labelcolor=INK, fontsize=9)
fig.savefig(uos.tagged("figures/beam_map.png"), dpi=200, bbox_inches="tight",
            facecolor=SURFACE)
plt.close(fig)
print("Saved", uos.tagged("figures/beam_map.png"))

# Loss of pointing straight at the user
fig, ax = base_axes(f"Loss of pointing straight at the user [dB], {uos.FREQ_GHZ:g} GHz")
im = ax.imshow(np.where(covered, np.minimum(loss_db, 30), np.nan),
               origin="lower", extent=extent, cmap=loss_cmap, vmin=0,
               vmax=30, interpolation="nearest", zorder=2)
overlay(ax)
colorbar(fig, ax, im, "Best beam - geometric beam [dB]")
ax.legend(loc="lower left", frameon=False, labelcolor=INK, fontsize=9)
fig.savefig(uos.tagged("figures/geometric_beam_loss.png"), dpi=200, bbox_inches="tight",
            facecolor=SURFACE)
plt.close(fig)
print("Saved", uos.tagged("figures/geometric_beam_loss.png"))

# ------------------------------------------------------------ 3D view
# Trick: reuse the last Sionna radio map, but feed it the best-beam angle
# instead of the path gain, so the renderer paints it on the 3D scene.
assert rm_last is not None
beam_val = np.where(covered, angles[best] + MAX_ANGLE + 1.0, 0.0)  # 0 = no data
per_tri = np.concatenate([beam_val.ravel(), beam_val.ravel()]).astype(np.float32)
beam_tensor = mi.TensorXf(per_tri[None, :])


class _BeamMap(type(rm_last)):
    @property
    def path_gain(self):
        return beam_tensor


rm_beams = copy.copy(rm_last)
rm_beams.__class__ = _BeamMap
cam = Camera(position=mi.Point3f(cen.x - 550, cen.y - 650, 420),
             look_at=mi.Point3f(cen.x, cen.y, 40))
scene.render_to_file(camera=cam, filename=str(uos.tagged("figures/beam_map_3d.png")),
                     radio_map=rm_beams, rm_db_scale=False, rm_vmin=0.5,
                     rm_vmax=2 * MAX_ANGLE + 1.0, rm_cmap=beam_cmap,
                     resolution=(1600, 1000))
print("Saved", uos.tagged("figures/beam_map_3d.png"))
