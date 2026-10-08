"""
Step 1 - 3D digital twin of the University of Seoul campus (SceneBaker).

Downloads buildings, roads and vegetation from OpenStreetMap plus SRTM
terrain, builds a Sionna RT scene and renders two preview images.

Run from ~/Desktop/uos-beam-twin with the environment active:
    python 01_campus_scene.py
Outputs: scenes/uos/..., figures/campus_3d.png, figures/campus_top.png
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "sionna-scene-baker"))
import mitsuba as mi                   # noqa: E402
import requests                        # noqa: E402
import scenebaker                      # noqa: E402
from sionna.rt import Camera           # noqa: E402

# Area: UOS campus + neighbourhood to the north + apartments to the south
# + the foot of Mt. Baebong to the east
LAT_MIN, LAT_MAX = 37.5790, 37.5885
LON_MIN, LON_MAX = 127.0515, 127.0655
OUT_DIR = Path("scenes/uos")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# OpenStreetMap (Overpass) servers: if one is overloaded, try the next one
OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

t0 = time.time()
scene = None
for url in OVERPASS_SERVERS:
    osm_file = OUT_DIR / "osm" / "map.osm"
    cached = osm_file.exists() and osm_file.stat().st_size > 1000
    try:
        print("Trying server:", url if not cached else "(using cached data)")
        scene = scenebaker.load_scene(
            LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
            terrain=True,           # real terrain (the hill to the east)
            out_dir=OUT_DIR,
            reuse_osm=cached,       # do not download again if cached
            overpass_url=url,
        )
        break
    except requests.exceptions.RequestException as e:
        print("  Failed:", e)
if scene is None:
    sys.exit("No server answered. Wait a few minutes and try again.")
print(f"Scene built in {time.time() - t0:.1f} s")
print("Objects in the scene:", len(scene.objects))

x_min, x_max, y_min, y_max = scenebaker.bounds(scene)
cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
print(f"Size: {x_max - x_min:.0f} m (E-W) x {y_max - y_min:.0f} m (N-S)")
print("Ground height at the centre:", scenebaker.height(scene, cx, cy), "m")

# Oblique 3D view (from the south-west)
cam_3d = Camera(position=mi.Point3f(cx - 700, cy - 700, 450),
                look_at=mi.Point3f(cx, cy, 20))
scene.render_to_file(camera=cam_3d, filename=str(FIG_DIR / "campus_3d.png"),
                     resolution=(1600, 1000))
print("Saved figures/campus_3d.png")

# Near-vertical top view
cam_top = Camera(position=mi.Point3f(cx, cy - 60, 1400),
                 look_at=mi.Point3f(cx, cy, 0))
scene.render_to_file(camera=cam_top, filename=str(FIG_DIR / "campus_top.png"),
                     resolution=(1400, 1400))
print("Saved figures/campus_top.png")
