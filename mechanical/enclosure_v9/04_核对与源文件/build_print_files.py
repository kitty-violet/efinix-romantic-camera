"""Build real CAD solids from the accepted V9 dimensions, then verify exports."""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import struct
import time
import zipfile

import cadquery as cq
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
LAYOUT = SCRIPT_DIR / "layout_v9.json" if (SCRIPT_DIR / "layout_v9.json").exists() else PROJECT / "hardware/layout_v9.json"
OUT = SCRIPT_DIR.parent if (SCRIPT_DIR / "layout_v9.json").exists() else PROJECT.parent / "3D打印交付_相机外壳_V9"
H = json.loads(LAYOUT.read_text(encoding="utf-8"))
E, P = H["enclosure"], H["mainboard"]
C, T = E["camera"], E["camera"]["cap"]
V = cq.Vector
START = time.time()
SEAM_Y = 15.0
FASTENERS = [(x, z) for x in (-175.0, 175.0) for z in (-115.0, 115.0)]
REPORT = {"version": "V9-P1", "units": "mm", "parts": [], "checks": {}}


def log(message):
    print(f"[{time.time() - START:6.1f}s] {message}", flush=True)


def box(w, d, h, x=0, y=0, z=0):
    return cq.Workplane("XY").box(w, d, h).val().translate((x, y, z))


def cylinder(radius, height, x=0, y=0, z=0, axis="z"):
    direction = {"x": V(1, 0, 0), "y": V(0, 1, 0), "z": V(0, 0, 1)}[axis]
    return cq.Solid.makeCylinder(radius, height, V(x, y, z) - direction * (height / 2), direction)


def ring(outer, inner, height, x=0, y=0, z=0, axis="z"):
    return cylinder(outer, height, x, y, z, axis).cut(cylinder(inner, height + 0.2, x, y, z, axis))


def mapped(x, z, axis):
    if axis == "y":
        return V(x, 0, z)
    if axis == "x":
        return V(0, x, z)
    return V(x, z, 0)


def rect_wire(w, h, r=0, cx=0, cy=0, axis="z"):
    """Use the same quadratic corner curves as Three.js rect()."""
    if not r:
        return cq.Wire.makePolygon([mapped(cx + x, cy + y, axis) for x, y in
                                    [(-w / 2, -h / 2), (w / 2, -h / 2),
                                     (w / 2, h / 2), (-w / 2, h / 2)]], close=True)
    edges = []
    def point(x, y):
        return mapped(cx + x, cy + y, axis)
    def line(a, b):
        if math.dist(a, b) > 1e-8:
            edges.append(cq.Edge.makeLine(point(*a), point(*b)))
    def bezier(a, b, c):
        edges.append(cq.Edge.makeBezier([point(*a), point(*b), point(*c)]))
    x, y = w / 2, h / 2
    line((-x + r, -y), (x - r, -y))
    bezier((x - r, -y), (x, -y), (x, -y + r))
    line((x, -y + r), (x, y - r))
    bezier((x, y - r), (x, y), (x - r, y))
    line((x - r, y), (-x + r, y))
    bezier((-x + r, y), (-x, y), (-x, y - r))
    line((-x, y - r), (-x, -y + r))
    bezier((-x, -y + r), (-x, -y), (-x + r, -y))
    return cq.Wire.assembleEdges(edges)


def circle_wire(radius, x=0, y=0, axis="z"):
    normal = {"x": V(1, 0, 0), "y": V(0, -1, 0), "z": V(0, 0, 1)}[axis]
    return cq.Wire.makeCircle(radius, mapped(x, y, axis), normal)


def hex_wire(af, x=0, y=0, axis="z"):
    radius = af / math.sqrt(3)
    return cq.Wire.makePolygon([mapped(x + radius * math.cos(math.pi / 6 + i * math.pi / 3),
                                      y + radius * math.sin(math.pi / 6 + i * math.pi / 3), axis)
                                for i in range(6)], close=True)


def plate(wire, depth, x=0, y=0, z=0, axis="z", holes=()):
    direction = {"x": V(depth, 0, 0), "y": V(0, depth, 0), "z": V(0, 0, depth)}[axis]
    return cq.Solid.extrudeLinear(wire, list(holes), direction).translate(V(x, y, z) - direction * 0.5)


def rrplate(w, h, depth, x=0, y=0, z=0, axis="z", radius=0, holes=()):
    return plate(rect_wire(w, h, radius, axis=axis), depth, x, y, z, axis, holes)


def board_xy(u, v):
    return E["boardCenter"][0] + v - 50, E["boardCenter"][1] + u - 40


def fuse(shapes, label):
    combined = shapes[0].fuse(*shapes[1:])
    result = combined.clean()
    if not result.isValid() and combined.isValid():
        result = combined
    if not result.isValid():
        result = result.fix()
    if not result.isValid():
        raise ValueError(f"{label}: invalid fused BRep")
    log(f"{label}: {len(result.Solids())} solid(s), volume {result.Volume():.1f} mm3")
    return result


def vent_holes(face, center_y=80, center_z=0):
    spec = E["ventilation"]["rear" if face == "rear" else "sides"]
    lo, hi, pitch = (spec[k] for k in
                     (("xMin", "xMax", "pitchX") if face == "rear" else ("yMin", "yMax", "pitchY")))
    result = []
    for horizontal in np.arange(lo, hi + 0.01, pitch):
        for z in np.arange(spec["zMin"], spec["zMax"] + 0.01, spec["pitchZ"]):
            if face == "left" and any(
                abs(horizontal - p["y"]) < (spec["slot"][0] + p["width"]) / 2 + 3 and
                abs(z - p["z"]) < (spec["slot"][1] + p["height"]) / 2 + 3
                for p in (E["typeCWindow"], E["rj45Window"])
            ):
                continue
            axis = "y" if face == "rear" else "x"
            result.append(rect_wire(*spec["slot"], 1.2,
                                    horizontal - (0 if face == "rear" else center_y), z - center_z, axis))
    REPORT["checks"][f"{face}VentCount"] = len(result)
    return result


def build_walls():
    left_holes = vent_holes("left")
    for p in (E["typeCWindow"], E["rj45Window"]):
        left_holes.append(rect_wire(p["width"], p["height"], 1.5, p["y"] - 80, p["z"], "x"))
    exit_spec = E["displayHost"]["powerExit"]
    walls = [
        rrplate(336, 216, 3, y=-1.5, axis="y", radius=3,
                holes=[rect_wire(324, 204, 2, axis="y")]),
        rrplate(160, 216, 3, -166.5, 80, 0, "x", holes=left_holes),
        rrplate(160, 212.4, 3, 166.5, 80, -1.8, "x", holes=vent_holes("right", 80, -1.8)),
        box(3, 22, 2.8, 166.5, 11, 106.6),
        rrplate(330, 160, 3, 0, 80, -106.5,
                holes=[rect_wire(exit_spec["width"], exit_spec["depth"], 2, exit_spec["x"], exit_spec["y"] - 80)]),
        rrplate(336, 216, 3, 0, 161.5, 0, "y", 3,
                [circle_wire(18, axis="y"), *vent_holes("rear")]),
        box(101, 160, 3, -114.5, 80, 106.5),
        box(37, 22, 3, 146.5, 11, 106.5),
        box(192, 22, 3, 32, 11, 106.5),
        box(37, 133, 1.6, 146.5, 91.5, 103.65),
    ]
    # The preview showed edge ribs; add retaining lips and matching lid rebates.
    for y in (22.4, 160.6):
        walls.append(box(229, 1, 2, 50.5, y, 106))
        lip_y = y + (1 if y < 80 else -1)
        walls.append(box(229, 2.7, 1, 50.5, lip_y, 106.45))
    return fuse(walls, "Shell walls and sliding tracks")


def round_path(w, h, radius):
    x, z = w / 2, h / 2
    a, b, s = x - radius, z - radius, radius / math.sqrt(2)
    edges = []
    def line(p, q):
        edges.append(cq.Edge.makeLine(mapped(*p, "y"), mapped(*q, "y")))
    def arc(p, q, r):
        edges.append(cq.Edge.makeThreePointArc(mapped(*p, "y"), mapped(*q, "y"), mapped(*r, "y")))
    line((-a, z), (a, z))
    arc((a, z), (a + s, b + s), (x, b))
    line((x, b), (x, -b))
    arc((x, -b), (a + s, -b - s), (a, -z))
    line((a, -z), (-a, -z))
    arc((-a, -z), (-a - s, -b - s), (-x, -b))
    line((-x, -b), (-x, b))
    arc((-x, b), (-a - s, b + s), (-a, z))
    return cq.Wire.assembleEdges(edges)


def build_grip():
    spec = E["grip"]
    w, h = spec["centerline"]
    plane = cq.Plane(origin=(-w / 2 + spec["cornerRadius"], 0, h / 2),
                     xDir=(0, 0, 1), normal=(-1, 0, 0))
    # Quadratic corners match V9; spline follows the same half-ellipse finger groove.
    cross = cq.Workplane(plane).moveTo(-17, -3).lineTo(15, -3)
    cross = cross.bezier([(17, -3), (17, -1)], includeCurrent=True)
    cross = cross.lineTo(17, 13).bezier([(17, 15), (15, 15)], includeCurrent=True).lineTo(10.5, 15)
    cross = cross.spline([(10.5 * math.cos(math.pi * i / 48), 15 - 8 * math.sin(math.pi * i / 48))
                          for i in range(1, 49)], includeCurrent=True)
    cross = cross.lineTo(-17, 15).bezier([(-19, 15), (-19, 13)], includeCurrent=True)
    cross = cross.lineTo(-19, -1).bezier([(-19, -3), (-17, -3)], includeCurrent=True).close()
    grip = cross.sweep(round_path(w, h, spec["cornerRadius"]),
                       isFrenet=False, transition="transformed").val()
    loops = []
    for cx, y, z in E["cableLoops"]["centers"]:
        half, radius, top = 13, 6, z + 18
        edges = [
            cq.Edge.makeLine(V(cx - half, y, z + 2), V(cx - half, y, top - radius)),
            cq.Edge.makeThreePointArc(V(cx - half, y, top - radius),
                                     V(cx - half + radius - radius / math.sqrt(2), y, top - radius + radius / math.sqrt(2)),
                                     V(cx - half + radius, y, top)),
            cq.Edge.makeLine(V(cx - half + radius, y, top), V(cx + half - radius, y, top)),
            cq.Edge.makeThreePointArc(V(cx + half - radius, y, top),
                                     V(cx + half - radius + radius / math.sqrt(2), y, top - radius + radius / math.sqrt(2)),
                                     V(cx + half, y, top - radius)),
            cq.Edge.makeLine(V(cx + half, y, top - radius), V(cx + half, y, z + 2)),
        ]
        path = cq.Wire.assembleEdges(edges)
        loop = cq.Workplane("XY", origin=(cx - half, y, z + 2)).circle(3).sweep(path, isFrenet=True).val()
        loops.extend([loop, box(10, 14, 8, cx - half, y, z), box(10, 14, 8, cx + half, y, z)])
    return fuse([grip, *loops], "Solid grip and strap eyes")


def diagonal(spec):
    a, b = V(*spec["start"]), V(*spec["end"])
    direction = (b - a).normalized()
    axis = V(0, 0, 1).cross(direction)
    angle = math.degrees(math.acos(V(0, 0, 1).dot(direction)))
    return box(*spec["section"], (b - a).Length).rotate((0, 0, 0), axis, angle).translate((a + b) * 0.5)


def build_supports():
    tray = E["tray"]
    holes = [rect_wire(w, d, 2, x, y - tray["center"][1])
             for w, d, x, y in tray["openings"]]
    for u, v in P["mountHoles"]:
        x, y = board_xy(u, v)
        holes.append(circle_wire(1.7, x, y - tray["center"][1]))
    w, d, x, y = tray["ffcSlot"]
    holes.append(rect_wire(w, d, 1, x, y - tray["center"][1]))
    shapes = [rrplate(tray["width"], tray["length"], 2, *tray["center"], radius=2, holes=holes)]
    for spec in P["mountHoleSpecs"]:
        x, y = board_xy(spec["u"], spec["v"])
        low, high = spec["nutPocketBottomZ"], 83.2
        shapes.append(ring(4.5, 1.7, low - 79, x, y, (79 + low) / 2))
        shapes.append(plate(circle_wire(4.5), high - low, x, y, (low + high) / 2,
                            holes=[hex_wire(spec["nutPocketAcrossFlats"])]))
    for x in tray["supportX"]:
        shapes.extend([box(6, 96, 4, x, 109, 75), box(6, 8, 56, x, 156, 49),
                       box(12, 4, 12, x, 158, 25)])
    shapes.extend(diagonal(spec) for spec in E["braces"])
    bx, by = board_xy(*H["bluetooth"]["localCenterUVZ"][:2])
    for x in (bx - 22.5, bx + 22.5):
        shapes.append(box(3, 28, 13.2, x, by, 85.6))
    return shapes


def thread_tooth(root, crest):
    # Map a left-hand Z helix onto world +Y so phase matches V9 theta=atan2(z,x).
    path = cq.Wire.makeHelix(T["pitch"], 12, root, lefthand=True)
    tooth = cq.Workplane("XZ").polyline([
        (root - 0.15, -0.88), (crest, -0.32), (crest, 0.32), (root - 0.15, 0.88)
    ]).close().sweep(path, isFrenet=True).val()
    return tooth.rotate((0, 0, 0), (1, 0, 0), -90).translate((0, 172, 0))


def build_camera():
    base = ring(24, 18, 3, y=164.5, axis="y")
    barrel = ring(21, 18, 16, y=174, axis="y")
    holes = [rect_wire(14, 3, 0.5, 0, -12.5, "y")]
    holes.extend(circle_wire(1.3, x, z, "y") for x, z in C["fastenerCentersXZ"])
    mount = plate(circle_wire(21, axis="y"), 4, y=184, axis="y", holes=holes)
    # Avoid an exactly tangent cylinder/helix Boolean; the tooth itself still ends at radius22.
    envelope = cq.Solid.makeCone(21, 22.01, 0.6, V(0, 174, 0), V(0, 1, 0)).fuse(
        cylinder(22.01, 6.8, y=178, axis="y"),
        cq.Solid.makeCone(22.01, 21, 0.6, V(0, 181.4, 0), V(0, 1, 0)))
    male = thread_tooth(21, 22).intersect(envelope)
    if male.Volume() < 400:
        raise ValueError("CAD Boolean lost the external helical teeth.")
    stop = ring(25, 20.9, 1.5, y=171.75, axis="y")
    camera = fuse([base, barrel, mount, male, stop], "Hollow camera cylinder and helical male thread")
    # The loose cap's front projection is 13 from y=186, independent of its rear skirt.
    outer = cylinder(25, 23, y=184, axis="y").fuse(
        cq.Solid.makeCone(25, 24, 3.5, V(0, 195.5, 0), V(0, 1, 0)))
    cap = outer.cut(cylinder(8, 40, y=185, axis="y"))
    cap = cap.cut(cylinder(22.36, 15.5, y=189.75, axis="y"))  # y182..197.5
    cap = cap.cut(cylinder(22.36, 1.6, y=173.2, axis="y"))  # rear lead-in y172.4..174
    cap = cap.cut(cylinder(21.35, 8.2, y=178, axis="y"))
    female_tool = thread_tooth(21.35, 22.35).intersect(box(60, 8, 60, y=178))
    cap = cap.cut(female_tool).clean()
    ribs = []
    for i in range(48):
        angle = i * 360 / 48
        ribs.append(box(0.65, 6.5, 0.6, 25, T["rearY"] + 4, 0)
                    .rotate((0, 0, 0), (0, 1, 0), -angle))
    cap = fuse([cap, *ribs], "Independent female-thread camera cap")
    return camera, cap, male


def build_lid():
    holes = [rect_wire(12, 2.8, 1.2, x, y)
             for x in np.arange(-78, 78.01, 19.5) for y in np.arange(-49, 49.01, 14)]
    lid = rrplate(232, 136.4, 2.4, 52, 91.5, 106.8, radius=1, holes=holes)
    for y in (24.35, 158.65):
        lid = lid.cut(box(233, 2.9, 1.65, 52, y, 106.425))
    additions = [lid]
    for y in (24.35, 158.65):
        additions.append(box(232, 2.4, 1, 52, y, 105.2))
        # A web connects the tongue to the panel, behind the retaining lip.
        additions.append(box(232, 0.65, 2.7, 52, y + (0.95 if y < 80 else -0.95), 106.05))
    for x in (-52, -49, -46):
        additions.append(box(1, 16, 0.45, x, 91.5, 108.225))
    return fuse(additions, "Removable ventilated sliding lid")


def add_split_joint(walls, grip, supports, camera):
    front_blank = fuse([walls.intersect(box(1000, 18, 1000, y=6)), grip], "Front split blank")
    rear_blank = walls.intersect(box(1000, 400, 1000, y=215))
    front_shapes, rear_shapes = [front_blank], [rear_blank, camera, *supports]
    for x, z in FASTENERS:
        front_shapes.append(cylinder(6, 17.8, x, 6.1, z, "y"))
        # Rounded tabs sit behind the existing grip, outside the screen envelope.
        rear_shapes.append(rrplate(22, 22, 8, math.copysign(172, x), 19,
                                   math.copysign(112, z), "y", 3))
    front = fuse(front_shapes, "Front frame joint pads")
    rear = fuse(rear_shapes, "Rear case, tray, braces, camera and joint tabs")
    rear = rear.cut(box(330, 5, 210, y=17.5))
    for x, z in FASTENERS:
        tool = cylinder(1.7, 45, x, 10, z, "y")
        front, rear = front.cut(tool), rear.cut(tool)
        rear = rear.cut(plate(hex_wire(5.9, axis="y"), 2.8, x, 21.7, z, "y"))
    # Small rear lands retain the screen against the front lip; host perimeter stays clear.
    lands = []
    for x in (-156, 156):
        for z in (-94, 94):
            lands.append(box(18, 4, 10, x, 22, z))
    rear = fuse([rear, *lands], "Rear screen-retaining lands")
    return front.clean(), rear.clean()


def bounds(shape):
    b = shape.BoundingBox()
    return {"min": [b.xmin, b.ymin, b.zmin], "max": [b.xmax, b.ymax, b.zmax],
            "size": [b.xlen, b.ylen, b.zlen]}


def require_single(shape, label):
    if not shape.isValid() or len(shape.Solids()) != 1 or not shape.Solids()[0].isValid():
        raise ValueError(f"{label} must be exactly one valid solid; got {len(shape.Solids())}")
    return shape.Solids()[0]


def validate_stl(path):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * count:
        raise ValueError(f"{path.name}: malformed binary STL")
    records = np.frombuffer(data, dtype=np.dtype([("normal", "<f4", (3,)),
                            ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]), offset=84, count=count)
    triangles = records["vertices"].astype(np.float64)
    if not np.isfinite(triangles).all():
        raise ValueError(f"{path.name}: non-finite mesh coordinates")
    vertices, inverse = np.unique(np.round(triangles.reshape(-1, 3), 5), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, edge_counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    volume = np.einsum("ij,ij->i", triangles[:, 0],
                       np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
    degenerate = int(np.count_nonzero(area < 1e-10))
    boundary = int(np.count_nonzero(edge_counts == 1))
    nonmanifold = int(np.count_nonzero(edge_counts > 2))
    # Directed pairs must cancel across every edge, not merely have count two.
    direction = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    sorted_edges, edge_ids = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    winding = np.bincount(edge_ids, weights=direction, minlength=len(sorted_edges))
    winding_errors = int(np.count_nonzero(winding))
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    used_vertices, used_faces = np.unique(faces, return_inverse=True)
    compact = used_faces.reshape(-1, 3)
    graph_edges = np.concatenate([compact[:, [0, 1]], compact[:, [1, 2]], compact[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(graph_edges)), (graph_edges[:, 0], graph_edges[:, 1])),
                       shape=(len(used_vertices), len(used_vertices))).tocsr()
    components = connected_components(graph, directed=False, return_labels=False)
    if boundary or nonmanifold or winding_errors or degenerate or volume <= 0 or components != 1:
        raise ValueError(f"{path.name}: STL invalid: boundary={boundary}, nonmanifold={nonmanifold}, "
                         f"winding={winding_errors}, degenerate={degenerate}, volume={volume}, components={components}")
    return {"triangles": count, "weldedVertices": len(vertices), "boundaryEdges": boundary,
            "nonManifoldEdges": nonmanifold, "windingErrors": winding_errors,
            "degenerateTriangles": degenerate, "signedVolume": volume, "watertight": True,
            "connectedComponents": int(components),
            "min": triangles.min(axis=(0, 1)).tolist(), "max": triangles.max(axis=(0, 1)).tolist()}


def normalize_stl(path):
    """Weld coincident float vertices and remove mesher's zero-area pole triangles."""
    data = path.read_bytes()
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    records = np.frombuffer(data, dtype=dtype, offset=84)
    vertices, inverse = np.unique(np.round(records["vertices"].reshape(-1, 3).astype(np.float64), 5),
                                  axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    points = vertices[faces]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    valid = np.linalg.norm(normals, axis=1) > 2e-10
    faces = faces[valid]
    canonical = np.sort(faces, axis=1)
    _, unique = np.unique(canonical, axis=0, return_index=True)
    faces = faces[np.sort(unique)]
    points = vertices[faces].astype(np.float32)
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError(f"{path.name}: zero triangle after float32 conversion")
    normals /= lengths[:, None]
    cleaned = np.zeros(len(faces), dtype=dtype)
    cleaned["normal"], cleaned["vertices"] = normals, points
    header = b"V9-P1 millimeters; welded closed triangle mesh".ljust(80, b"\0")
    path.write_bytes(header + struct.pack("<I", len(cleaned)) + cleaned.tobytes())
    return {"removedPoleOrDuplicateTriangles": len(records) - len(cleaned), "coordinateWeldToleranceMM": 0.00001}


def export_part(shape, stem, label, quantity=1, category="01_正式打印零件"):
    shape = require_single(shape, label)
    folder = OUT / category
    folder.mkdir(parents=True, exist_ok=True)
    b = bounds(shape)
    translation = [-v for v in b["min"]]
    printable = shape.translate(translation)
    stl_path, step_path = folder / f"{stem}.stl", folder / f"{stem}.stp"
    cq.exporters.export(printable, str(step_path), exportType="STEP")
    printable.exportStl(str(stl_path), tolerance=0.025, angularTolerance=0.08, relative=False)
    normalization = normalize_stl(stl_path)
    checked = validate_stl(stl_path)
    reimported = cq.importers.importStep(str(step_path)).val()
    require_single(reimported, f"STEP readback {label}")
    volume_error = abs(checked["signedVolume"] - shape.Volume()) / shape.Volume()
    if volume_error > 0.004:
        raise ValueError(f"{label}: STL tessellation changes volume by {volume_error}")
    part = {"id": stem, "label": label, "quantity": quantity, "category": category,
            "stl": str(stl_path.relative_to(OUT)).replace("\\", "/"),
            "step": str(step_path.relative_to(OUT)).replace("\\", "/"),
            "assemblyBounds": b, "printToAssemblyTranslation": b["min"],
            "volumeMM3": shape.Volume(), "stlVolumeRelativeError": volume_error,
            "brepValid": True, "solidCount": 1, "stepReadbackValid": True,
            "meshNormalization": normalization, "mesh": checked}
    REPORT["parts"].append(part)
    log(f"Exported {stem}: {count_text(checked['triangles'])} triangles; closed mesh and STEP readback passed")
    return part


def count_text(value):
    return f"{value:,}"


def verify_assembly(front, rear, lid, cap, camera):
    checks = REPORT["checks"]
    tooth_samples = []
    for y in (175, 176, 177, 178, 179, 180, 181):
        expected_male = int(y) % 2 == 0
        actual_male = camera.isInside(V(21.5, y, 0))
        actual_female = cap.isInside(V(21.5, y, 0))
        tooth_samples.append({"y": y, "radius": 21.5, "maleSolid": actual_male, "femaleSolid": actual_female})
        if actual_male != expected_male or actual_female == expected_male:
            raise ValueError("Exported CAD must contain alternating male/female helical teeth.")
    checks["actualHelicalToothSamples"] = tooth_samples
    for name, a, b in [("frameToRear", front, rear), ("lidToRear", lid, rear),
                       ("capToCamera", cap, camera)]:
        overlap = a.intersect(b).Volume()
        checks[f"{name}OverlapMM3"] = overlap
        if overlap > 0.005:
            raise ValueError(f"{name}: interfering volume {overlap} mm3")
    lid_samples = []
    for travel in (0, 10, 50, 100, 150, 196):
        overlap = lid.translate((travel, 0, 0)).intersect(rear).Volume()
        lid_samples.append({"travel": travel, "intersectionMM3": overlap})
        if overlap > 0.005:
            raise ValueError(f"Lid obstructed at {travel}: {overlap}")
    checks["lidMotion"] = lid_samples
    screw_samples = []
    for travel in (0, 0.5, 1, 2, 4, 6, 8, 10):
        displaced = cap.rotate((0, 0, 0), (0, 1, 0), -travel / T["pitch"] * 360).translate((0, travel, 0))
        overlap = displaced.intersect(camera).Volume()
        screw_samples.append({"travel": travel, "rotationDegrees": -travel / T["pitch"] * 360,
                              "intersectionMM3": overlap})
        if overlap > 0.005:
            raise ValueError(f"Thread interferes at {travel}: {overlap}")
    checks["capScrewMotion"] = screw_samples
    passage_checks = []
    for x, z in C["fastenerCentersXZ"]:
        test = cylinder(1.2, 4.4, x, 184, z, "y")
        overlap = test.intersect(rear).Volume()
        passage_checks.append(overlap)
        if overlap > 1e-5:
            raise ValueError("Camera mounting hole is blocked.")
        driver = cylinder(3, 180.2, x, 88.1, z, "y")  # y=-2..178.2
        if driver.intersect(rear).Volume() > 1e-5:
            raise ValueError("Inner camera nut driver access is blocked.")
    for u, v in P["mountHoles"]:
        x, y = board_xy(u, v)
        if cylinder(1.6, 6.8, x, y, 80.1).intersect(rear).Volume() > 1e-5:
            raise ValueError("Mainboard through bore is blocked.")
    if box(13, 4.4, 2, 0, 184, -12.5).intersect(rear).Volume() > 1e-5:
        raise ValueError("Lower FPC slit is blocked.")
    if not rear.isInside(V(0, 184, 0)):
        raise ValueError("Old central FPC opening is not closed.")
    # Screen is inserted through the open split, then captured by the front lip and rear lands.
    screen = box(329.99, 19.99, 209.99, y=10)
    for name, printed in [("rear", rear), ("front", front)]:
        collision = screen.intersect(printed)
        if collision.Volume() > 1e-4:
            raise ValueError(f"The screen envelope intersects {name}: "
                             f"{[(s.Volume(), bounds(s)) for s in collision.Solids()]}")
    for x, z in FASTENERS:
        if cylinder(1.6, 30, x, 10, z, "y").intersect(front).Volume() > 1e-5 or \
           cylinder(1.6, 30, x, 10, z, "y").intersect(rear).Volume() > 1e-5:
            raise ValueError("Front-frame joining screw passage blocked.")
    checks.update({"screenEnvelope": [330, 210, 20], "screenInsertionFromOpenFront": True,
                   "frameSplitY": SEAM_Y, "frameJoiningScrews": {"centersXZ": FASTENERS, "bore": 3.4, "nutAF": 5.9},
                   "cameraHolePassages": passage_checks, "innerCameraNutAccess": True,
                   "cameraFPCSlit": [14, 3], "cameraFPCSlitCenterXZ": [0, -12.5],
                   "cameraCapProjection": cap.BoundingBox().ymax - 186,
                   "cameraAperture": 16, "threadRadialAllowance": 0.35,
                   "displayPortStatus": "V9 photo-estimated positions retained; not remeasured"})


def save_report():
    (OUT / "04_核对与源文件").mkdir(parents=True, exist_ok=True)
    REPORT["elapsedSeconds"] = time.time() - START
    (OUT / "04_核对与源文件/模型与网格检查.json").write_text(
        json.dumps(REPORT, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "零件清单.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["零件编号", "零件名称", "每套数量", "X尺寸_mm", "Y尺寸_mm", "Z尺寸_mm", "STL文件", "STEP文件"])
        for part in REPORT["parts"]:
            writer.writerow([part["id"], part["label"], part["quantity"],
                             *[round(x, 3) for x in part["assemblyBounds"]["size"]], part["stl"], part["step"]])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "04_核对与源文件").mkdir(exist_ok=True)
    walls, grip = build_walls(), build_grip()
    camera, cap, male = build_camera()
    supports = build_supports()
    lid = build_lid()
    front, rear = add_split_joint(walls, grip, supports, camera)
    front, rear, lid, cap = [require_single(shape, name) for shape, name in
                              [(front, "front"), (rear, "rear"), (lid, "lid"), (cap, "cap")]]
    log("Checking real solid intersections, screw travel and screen insertion")
    verify_assembly(front, rear, lid, cap, camera)
    assembled = cq.Assembly(name="Camera_Enclosure_V9_P1")
    solids = [
        ("01_front_frame_grip", "前框_实心抓边_挂绳耳", front, (0.3, 0.36, 0.38)),
        ("02_rear_case_camera", "后壳_主板托架_摄像头螺纹筒", rear, (0.59, 0.62, 0.63)),
        ("03_sliding_lid", "顶部散热滑盖", lid, (0.39, 0.66, 0.62)),
        ("04_camera_screw_cap", "摄像头内螺纹罩盖", cap, (0.84, 0.65, 0.32)),
    ]
    for stem, label, shape, color in solids:
        export_part(shape, stem, label)
        assembled.add(shape, name=stem, color=cq.Color(*color))
        cq.exporters.export(shape, str(OUT / "04_核对与源文件" / f"{stem}_assembly.brep"), exportType="BREP")
    export_part(ring(1.8, 0.9, 1, y=186.5, axis="y"), "05_camera_spacer_1mm", "摄像头1mm隔套_可用金属件代替", 4)
    export_part(ring(2.7, 1.7, 7.4, z=88.5), "06_IO_spacer_7p4mm", "IO板7.4mm隔套_可用金属件代替", 2)
    assembled.export(str(OUT / "整机装配_V9_P1.stp"), exportType="STEP")
    # Matched small coupons isolate screw fit before the large case is printed.
    coupon_male = fuse([ring(21, 18, 10, y=178, axis="y"), male,
                       ring(25, 18, 1.5, y=172.25, axis="y")], "Male thread fit coupon")
    coupon_female = cap.intersect(box(60, 10.3, 60, y=177.65)).clean()
    export_part(coupon_male, "T1_male_thread_fit", "外螺纹试配环", category="02_先打螺纹试配件")
    export_part(coupon_female, "T2_female_thread_fit", "内螺纹试配环", category="02_先打螺纹试配件")
    for source, target in [(LAYOUT, OUT / "04_核对与源文件/layout_v9.json"),
                           (Path(__file__), OUT / "04_核对与源文件/build_print_files.py")]:
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    save_report()
    log("CAD and mesh export complete")


if __name__ == "__main__":
    main()
