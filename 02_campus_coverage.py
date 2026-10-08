"""
Step 2 - 5G base station on a campus rooftop and its coverage map.

- Uses the scene built in step 1 (scenes/uos).
- Places the gNB on the highest rooftop inside the campus.
- Computes coverage on a surface that follows the terrain (1.5 m above
  ground, i.e. handset height) instead of a flat plane.

Run from ~/Desktop/uos-beam-twin with the environment active:
    python 02_campus_coverage.py              (3.5 GHz, default)
    python 02_campus_coverage.py --freq 4.7   (4.7 GHz)
Outputs: figures/coverage_3d_<freq>.png, figures/coverage_top_<freq>.png
"""
import time
from pathlib import Path

import mitsuba as mi
from sionna.rt import Camera, PlanarArray, RadioMapSolver, Transmitter

import uos_twin as uos

CELL = 3.0            # map cell size [m]
MARGIN = 120.0        # map extent around the campus [m]
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- scene
t0 = time.time()
scene = uos.load_uos_scene()
campus = uos.campus_polygon()
print(f"Scene loaded in {time.time() - t0:.1f} s  |  carrier {uos.FREQ_GHZ:g} GHz")
print(f"Campus area: {campus.area / 1e4:.1f} ha")

# ------------------------------------------- highest campus rooftop -> gNB
tx_x, tx_y, roof_z, ground = uos.highest_roof(scene, campus)
tx_lat, tx_lon = uos.PROJ.unproject(tx_x, tx_y)
print(f"gNB at ({tx_x:.0f}, {tx_y:.0f}) m -> lat {tx_lat:.5f}, lon {tx_lon:.5f}")
print(f"  ground {ground:.1f} m, roof {roof_z:.1f} m, "
      f"antenna {roof_z + uos.MAST:.1f} m")

scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901",
                             polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                             polarization="V")
tx = Transmitter(name="gNB",
                 position=mi.Point3f(tx_x, tx_y, roof_z + uos.MAST),
                 display_radius=6)
scene.add(tx)
# Point the sector towards the campus centre (tilted down to the ground)
c = campus.centroid
tx.look_at(mi.Point3f(c.x, c.y, uos.ground_z(scene, c.x, c.y)))

# ----------------------------- measurement surface that follows the terrain
t0 = time.time()
xmin, ymin, xmax, ymax = campus.bounds
xs, ys, Z = uos.terrain_grid(scene, xmin - MARGIN, xmax + MARGIN,
                             ymin - MARGIN, ymax + MARGIN, CELL)
surface = uos.grid_to_mesh(xs, ys, Z)
print(f"Measurement surface: {surface.face_count()} triangles "
      f"({time.time() - t0:.1f} s)")

# ------------------------------------------------------------ ray tracing
t0 = time.time()
rm = RadioMapSolver()(scene=scene, measurement_surface=surface,
                      samples_per_tx=2 * 10**7, max_depth=5,
                      diffraction=True)
print(f"Coverage map computed in {time.time() - t0:.1f} s")

# ------------------------------------------------------------ images
cam_3d = Camera(position=mi.Point3f(c.x - 550, c.y - 650, 420),
                look_at=mi.Point3f(c.x, c.y, 40))
scene.render_to_file(camera=cam_3d, filename=str(uos.tagged("figures/coverage_3d.png")),
                     radio_map=rm, rm_vmin=-120, rm_vmax=-60,
                     resolution=(1600, 1000))
print("Saved", uos.tagged("figures/coverage_3d.png"))

cam_top = Camera(position=mi.Point3f(c.x, c.y - 40, 1100),
                 look_at=mi.Point3f(c.x, c.y, 0))
scene.render_to_file(camera=cam_top, filename=str(uos.tagged("figures/coverage_top.png")),
                     radio_map=rm, rm_vmin=-120, rm_vmax=-60,
                     resolution=(1400, 1400))
print("Saved", uos.tagged("figures/coverage_top.png"))
