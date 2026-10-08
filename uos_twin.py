"""
Shared helpers for the University of Seoul digital twin
(scene loading, campus outline, gNB placement, measurement grids).
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "sionna-scene-baker"))

import mitsuba as mi                                    # noqa: E402
from shapely.geometry import Point, Polygon             # noqa: E402
from sionna.rt import Scene, load_scene                 # noqa: E402
import scenebaker                                       # noqa: E402
from src.projection import LocalProjection             # noqa: E402
from src.scene_utils import register                    # noqa: E402

# Area used in 01_campus_scene.py (defines the scene coordinate origin)
LAT_MIN, LAT_MAX = 37.5790, 37.5885
LON_MIN, LON_MAX = 127.0515, 127.0655
SCENE_XML = ROOT / "scenes/uos/uos_scenebaker_terrain.xml"
OSM_FILE = ROOT / "scenes/uos/osm/map.osm"

# Carrier frequency, chosen on the command line: python <script>.py --freq 4.7
#   3.5 GHz: band n78, public 5G of the Korean operators (default)
#   4.7 GHz: band n79, Korean private 5G networks (e-Um 5G)
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--freq", type=float, default=3.5,
                     help="carrier frequency in GHz (default 3.5)")
FREQ_GHZ: float = _parser.parse_known_args()[0].freq
FREQ = FREQ_GHZ * 1e9
TAG = f"{FREQ_GHZ:g}GHz".replace(".", "p")          # e.g. "3p5GHz"
MAST = 3.0            # mast height above the roof [m]
UE_HEIGHT = 1.5       # handset height above ground [m]

PROJ = LocalProjection(lat0=(LAT_MIN + LAT_MAX) / 2,
                       lon0=(LON_MIN + LON_MAX) / 2)


def tagged(path: str, tag: str | None = None) -> Path:
    """Add the frequency tag to a file name:
    'figures/beam_map.png' -> 'figures/beam_map_3p5GHz.png'."""
    p = Path(path)
    return p.with_name(f"{p.stem}_{tag or TAG}{p.suffix}")


def load_uos_scene() -> Scene:
    scene = load_scene(str(SCENE_XML))
    register(scene, SCENE_XML)          # enables scenebaker.height(...)
    scene.frequency = FREQ
    return scene


def _osm():
    root = ET.parse(OSM_FILE).getroot()
    nodes = {n.get("id"): (float(n.get("lat", 0)), float(n.get("lon", 0)))
             for n in root.iter("node")}
    return root, nodes


def campus_polygon() -> Polygon:
    root, nodes = _osm()
    for way in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in way.iter("tag")}
        if tags.get("name:en") == "University of Seoul":
            return Polygon([PROJ.project(*nodes[nd.get("ref")])
                            for nd in way.iter("nd")])
    raise RuntimeError("Campus outline not found in the OSM file")


def building_footprints() -> list[np.ndarray]:
    """Building outlines (x, y), used to draw 2D maps."""
    root, nodes = _osm()
    out = []
    for way in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in way.iter("tag")}
        if "building" in tags:
            pts = [PROJ.project(*nodes[nd.get("ref")])
                   for nd in way.iter("nd") if nd.get("ref") in nodes]
            if len(pts) >= 3:
                out.append(np.array(pts))
    return out


WALKABLE = {"footway", "pedestrian", "path", "service", "residential",
            "living_street", "tertiary", "unclassified", "steps"}


def walkway_graph(area: Polygon):
    """Walkable OSM ways inside 'area' as a graph.
    Returns (xy, edges): node coordinates {id: (x, y)} and adjacency
    {id: [(neighbour_id, length_m), ...]}."""
    root, nodes = _osm()
    xy: dict[str, tuple[float, float]] = {}
    edges: dict[str, list[tuple[str, float]]] = {}
    for way in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in way.iter("tag")}
        if tags.get("highway") not in WALKABLE:
            continue
        refs = [nd.get("ref") for nd in way.iter("nd")]
        refs = [r for r in refs if r is not None and r in nodes]
        for a, b in zip(refs[:-1], refs[1:]):
            pa, pb = PROJ.project(*nodes[a]), PROJ.project(*nodes[b])
            if not (area.contains(Point(pa)) and area.contains(Point(pb))):
                continue
            d = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
            xy[a], xy[b] = pa, pb
            edges.setdefault(a, []).append((b, d))
            edges.setdefault(b, []).append((a, d))
    return xy, edges


def ground_z(scene: Scene, x: float, y: float) -> float:
    z = scenebaker.height(scene, x, y)
    return float(z) if z is not None else 0.0


def highest_roof(scene: Scene, area: Polygon, step: float = 5.0):
    """(x, y, z_roof, z_ground) of the highest rooftop inside 'area'."""
    xmin, ymin, xmax, ymax = area.bounds
    best = None
    for x in np.arange(xmin, xmax, step):
        for y in np.arange(ymin, ymax, step):
            if not area.contains(Point(x, y)):
                continue
            g = scenebaker.height(scene, x, y)
            r = scenebaker.height(scene, x, y, buildings=True)
            if g is None or r is None or r - g < 5.0:
                continue
            if best is None or r > best[2]:
                best = (float(x), float(y), float(r), float(g))
    if best is None:
        raise RuntimeError("No buildings found")
    return best


def terrain_grid(scene: Scene, x0, x1, y0, y1, cell, dz=UE_HEIGHT):
    """Regular grid that follows the terrain. Returns (xs, ys, Z)."""
    xs = np.arange(x0, x1 + cell, cell)
    ys = np.arange(y0, y1 + cell, cell)
    Z = np.array([[ground_z(scene, x, y) for x in xs] for y in ys]) + dz
    return xs, ys, Z


def grid_to_mesh(xs, ys, Z) -> mi.Mesh:
    """Measurement mesh: 2 triangles per cell.
    Triangles k and k + n_cells belong to the same cell k."""
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.stack([a, b, d], 1),
                            np.stack([a, d, c], 1)]).astype(np.uint32)
    mesh = mi.Mesh("measurement", vertex_count=len(verts), face_count=len(faces),
                   has_vertex_normals=False, has_vertex_texcoords=False)
    params = mi.traverse(mesh)
    params["vertex_positions"] = mi.Float(verts.ravel())
    params["faces"] = mi.UInt32(faces.ravel())
    params.update()
    return mesh


def rotation_matrix_np(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Same rotation as Sionna (3GPP TR 38.901, eq. 7.1-4).
    Columns are the local (x, y, z) axes expressed in world coordinates."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cc, sc = np.cos(gamma), np.sin(gamma)
    return np.array([
        [ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
        [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
        [-sb, cb * sc, cb * cc],
    ])
