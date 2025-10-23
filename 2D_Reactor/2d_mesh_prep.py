#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLB -> 2D silhouette -> 2D triangulated mesh (OBJ, y=0).

Dependencies (required): numpy, pillow, trimesh
Optional (recommended for tough shapes): mapbox_earcut, shapely

Usage:
  python glb_to_2d_silhouette_mesh.py --in jar.glb --outdir ./out \
      [--res 1024] [--view +y] [--margin 0.05] \
      [--eps 0.6] [--scale 1.0] [--aa 8] \
      [--close 1] [--tri auto|earcut|earclip] [--thresh 200]

Outputs:
  - silhouette.png   (orthographic front/side/top view)
  - vase2d.obj       (2D triangulated mesh in x–z plane; y=0)
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import trimesh
import numpy as np

# -----------------------------
# Mesh load + normalize
# -----------------------------
def load_align_normalize(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path.as_posix(), force='mesh')
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(g for g in m.geometry.values()))
    # cleanup (newer trimesh: avoid deprecation)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    m.merge_vertices()

    # align tallest principal axis to +Z
    v = m.vertices - m.center_mass
    cov = np.cov(v.T)
    w, U = np.linalg.eigh(cov)
    z_axis = U[:, np.argmax(w)]
    align_T = trimesh.geometry.align_vectors(z_axis, np.array([0.0, 0.0, 1.0]))
    m.apply_transform(align_T)

    # center + scale to unit sphere
    m.apply_translation(-m.center_mass)
    scale = 1.0 / max(1e-9, np.linalg.norm(m.vertices, axis=1).max())
    m.apply_scale(scale)
    return m

# -----------------------------
# Silhouette raster (orthographic)
# -----------------------------
def silhouette_raster(mesh: trimesh.Trimesh, res=512, margin=0.05, view='+y', aa=4, out_path: Path=None):
    # choose orthographic plane
    if view == '+y':   # look along -y ⇒ plane x–z
        u_axis = np.array([1.0, 0.0, 0.0])
        v_axis = np.array([0.0, 0.0, 1.0])
    elif view == '+x': # plane y–z
        u_axis = np.array([0.0, 1.0, 0.0])
        v_axis = np.array([0.0, 0.0, 1.0])
    elif view == '+z': # plane x–y
        u_axis = np.array([1.0, 0.0, 0.0])
        v_axis = np.array([0.0, 1.0, 0.0])
    else:
        raise ValueError("view must be +y, +x or +z")

    V = mesh.vertices; F = mesh.faces
    Uv = V @ u_axis; Vv = V @ v_axis
    umin, umax = Uv.min(), Uv.max()
    vmin, vmax = Vv.min(), Vv.max()
    du, dv = umax - umin, vmax - vmin
    umin -= margin * du; umax += margin * du
    vmin -= margin * dv; vmax += margin * dv

    R = res * aa
    img = Image.new('L', (R, R), 0)
    draw = ImageDraw.Draw(img)

    def to_px(u, v):
        x = (u - umin) / max(1e-12, (umax - umin)) * (R - 1)
        y = (1.0 - (v - vmin) / max(1e-12, (vmax - vmin))) * (R - 1)
        return x, y

    for tri in F:
        pts = [to_px(Uv[i], Vv[i]) for i in tri]
        draw.polygon(pts, fill=255)

    if aa > 1:
        img = img.resize((res, res), Image.Resampling.LANCZOS)
    if out_path is not None:
        img.save(out_path)
    return img

# -----------------------------
# Silhouette -> polygon helpers
# -----------------------------
def moore_trace(binary: np.ndarray):
    """Trace outer boundary of a binary blob (1=foreground) via Moore-Neighbor tracing."""
    B = np.pad((binary > 0).astype(np.uint8), 1, mode='constant')
    H, W = B.shape
    ys, xs = np.nonzero(B)
    if len(xs) == 0:
        return []
    idx = np.lexsort((xs, ys))[0]
    sy, sx = ys[idx], xs[idx]
    neighbors = [(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0)]
    py, px = sy, sx - 1
    y, x = sy, sx
    contour, first = [], True
    while True:
        # find neighbor index where previous lies
        start_k = 0
        for k,(dy,dx) in enumerate(neighbors):
            if (y+dy == py) and (x+dx == px):
                start_k = k; break
        found = False
        for i in range(8):
            k = (start_k + i) % 8
            dy, dx = neighbors[k]
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and B[ny, nx] == 1:
                py, px = y, x
                y, x = ny, nx
                found = True
                break
        contour.append((x - 1 + 0.5, y - 1 + 0.5))  # unpad + pixel-center
        if not first and (y == sy and x == sx and (py, px) == (sy, sx - 1)):
            break
        first = False
        if not found or len(contour) > H * W * 4:
            break
    if not contour:
        return []
    # remove immediate duplicates
    out = [contour[0]]
    for p in contour[1:]:
        if abs(p[0]-out[-1][0]) > 1e-6 or abs(p[1]-out[-1][1]) > 1e-6:
            out.append(p)
    return out

def rdp(points, eps):
    """Ramer–Douglas–Peucker on a closed polygon (epsilon in pixels)."""
    if len(points) < 3 or eps <= 0:
        return points[:]
    pts = np.array(points, dtype=np.float64)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    def perp_dist(p, a, b):
        p = np.array(p); a = np.array(a); b = np.array(b)
        if np.allclose(a, b): return np.linalg.norm(p - a)
        return abs(np.cross(b - a, a - p)) / (np.linalg.norm(b - a) + 1e-12)
    def simplify(seq):
        if len(seq) <= 2: return seq
        a, b = seq[0], seq[-1]
        dmax, idx = -1.0, -1
        for i in range(1, len(seq)-1):
            d = perp_dist(seq[i], a, b)
            if d > dmax: dmax, idx = d, i
        if dmax > eps:
            left = simplify(seq[:idx+1]); right = simplify(seq[idx:])
            return left[:-1] + right
        else:
            return [tuple(a), tuple(b)]
    res = simplify([tuple(p) for p in pts])
    if res[0] != res[-1]:
        res.append(res[0])
    return res

def remove_near_duplicates(poly, tol=1e-6):
    out = [poly[0]]
    for p in poly[1:]:
        if abs(p[0]-out[-1][0]) > tol or abs(p[1]-out[-1][1]) > tol:
            out.append(p)
    if out[0] != out[-1]:
        out.append(out[0])
    return out

def remove_collinear(poly, tol=1e-9):
    if poly[0] != poly[-1]:
        poly = poly + [poly[0]]
    clean = [poly[0]]
    for i in range(1, len(poly)-1):
        a = np.array(clean[-1]); b = np.array(poly[i]); c = np.array(poly[i+1])
        area2 = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        if abs(area2) > tol:
            clean.append(tuple(b))
    clean.append(clean[0])
    return clean

def area2(a,b,c): return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
def is_ccw(poly): return sum((poly[i+1][0]-poly[i][0])*(poly[i+1][1]+poly[i][1]) for i in range(len(poly)-1)) < 0
def point_in_tri(p,a,b,c):
    v0 = np.array(c)-np.array(a); v1=np.array(b)-np.array(a); v2=np.array(p)-np.array(a)
    den = v0[0]*v1[1]-v1[0]*v0[1]
    if abs(den) < 1e-12: return False
    u = (v2[0]*v1[1]-v1[0]*v2[1])/den
    v = (v0[0]*v2[1]-v2[0]*v0[1])/den
    return (u>=-1e-12) and (v>=-1e-12) and (u+v<=1+1e-12)

def earclip_triangulate(poly):
    if poly[0] != poly[-1]: poly = poly + [poly[0]]
    if not is_ccw(poly): poly = poly[::-1]
    V = list(range(len(poly)-1)); tris=[]
    it=0
    while len(V)>2 and it<10000:
        it+=1; ear=False; n=len(V)
        for i in range(n):
            i0=V[(i-1)%n]; i1=V[i]; i2=V[(i+1)%n]
            a,b,c = poly[i0], poly[i1], poly[i2]
            if area2(a,b,c) <= 0: continue
            inside=False
            for j in range(n):
                if j in [(i-1)%n, i, (i+1)%n]: continue
                p = poly[V[j]]
                if point_in_tri(p,a,b,c): inside=True; break
            if inside: continue
            tris.append((i0,i1,i2))
            del V[i]; ear=True; break
        if not ear: break
    return tris, poly[:-1]

# -----------------------------
# Binary post-process (optional)
# -----------------------------
def morph_close(bin_img, radius=1):
    """Tiny morphological closing without external deps (radius in pixels)."""
    if radius <= 0: return bin_img
    H, W = bin_img.shape
    # Dilate
    dil = np.zeros_like(bin_img, dtype=np.uint8)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            yy = np.clip(np.arange(H)[:,None] + dy, 0, H-1)
            xx = np.clip(np.arange(W)[None,:] + dx, 0, W-1)
            dil = np.maximum(dil, bin_img[yy, xx])
    # Erode
    ero = np.ones_like(bin_img, dtype=np.uint8)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            yy = np.clip(np.arange(H)[:,None] + dy, 0, H-1)
            xx = np.clip(np.arange(W)[None,:] + dx, 0, W-1)
            ero = np.minimum(ero, dil[yy, xx])
    return ero

# -----------------------------
# Export
# -----------------------------
def export_obj(verts2d, tris, outpath: Path, scale: float):
    with open(outpath.as_posix(), 'w') as f:
        for x,z in verts2d:
            f.write(f"v {x*scale:.6f} 0.000000 {z*scale:.6f}\n")
        for a,b,c in tris:
            f.write(f"f {a+1} {b+1} {c+1}\n")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='Input GLB/OBJ/PLY')
    ap.add_argument('--outdir', required=True, help='Output directory')
    ap.add_argument('--res', type=int, default=512, help='Silhouette resolution')
    ap.add_argument('--view', choices=['+y','+x','+z'], default='+y', help='Orthographic view')
    ap.add_argument('--margin', type=float, default=0.05, help='Silhouette frame margin')
    ap.add_argument('--eps', type=float, default=1.0, help='Polygon simplification epsilon (pixels)')
    ap.add_argument('--scale', type=float, default=1.0, help='Coordinate scale for OBJ')
    ap.add_argument('--aa', type=int, default=4, help='Antialiasing factor (supersampling)')
    ap.add_argument('--close', type=int, default=0, help='Morphological closing radius (0=off, 1..2 recommended)')
    ap.add_argument('--tri', choices=['auto','earcut','earclip'], default='auto', help='Triangulator to use')
    ap.add_argument('--thresh', type=int, default=200, help='Binarization threshold [0..255] (higher=cleaner edges)')
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    mesh = load_align_normalize(Path(args.inp))
    sil_path = outdir / "silhouette.png"
    img = silhouette_raster(mesh, res=args.res, margin=args.margin, view=args.view, aa=args.aa, out_path=sil_path)

    # Strict binary mask (reduce AA speckles)
    M = np.array(img.convert('L'))
    B = (M >= int(args.thresh)).astype(np.uint8)

    # Optional: seal tiny gaps
    if args.close > 0:
        B = morph_close(B, radius=int(args.close))

        # Strict binary mask (reduce AA speckles)
    M = np.array(img.convert('L'))
    B = (M >= int(args.thresh)).astype(np.uint8)

    # Optional: seal tiny gaps
    if args.close > 0:
        B = morph_close(B, radius=int(args.close))

    from skimage.measure import find_contours
    contours = find_contours(B, 0.5)
    if len(contours) == 0:
        raise SystemExit("No valid contour found in binary mask.")
    # pick the longest contour
    contour = max(contours, key=lambda c: len(c))
    # flip (row, col) -> (x, y)
    contour = np.fliplr(contour)
    print(f"[DEBUG] skimage found {len(contour)} contour points")



    contour = rdp(contour, eps=float(args.eps))
    contour = remove_near_duplicates(contour)
    contour = remove_collinear(contour)

    # Make polygon valid (Shapely optional)
    try:
        from shapely.geometry import Polygon, LinearRing
        ring = LinearRing(contour[:-1] if contour[0]==contour[-1] else contour)
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            raise ValueError("Empty polygon after validity fix.")
        if poly.geom_type == "MultiPolygon":
            poly = max(list(poly.geoms), key=lambda g: g.area)
        contour = list(poly.exterior.coords)
    except Exception:
        pass

    # -------- TRIANGULATE (uses args; MUST be inside main()) --------
    # Ensure contour is a clean Nx2 array (drop z if present)
    contour = np.asarray(contour, dtype=np.float64)
    if contour.ndim != 2:
        raise ValueError(f"Contour array has invalid shape {contour.shape}")
    if contour.shape[1] > 2:
        contour = contour[:, :2]  # drop z-coordinate if 3D
    elif contour.shape[1] < 2:
        raise ValueError(f"Contour has less than 2 dimensions: {contour.shape}")

    def tri_earclip(loop):
        t, v = earclip_triangulate(loop)
        return t, v

    def tri_earcut(loop):
        import mapbox_earcut as earcut

        loop2 = loop[:-1] if np.allclose(loop[0], loop[-1]) else loop
        pts = np.asarray(loop2, dtype=np.float64)

        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Invalid shape for vertices: {pts.shape}")

        # Force contiguous arrays
        verts = np.ascontiguousarray(pts, dtype=np.float64)
        holes = np.ascontiguousarray(np.array([len(verts)], dtype=np.uint32))  # 👈 add ring end index

        idx = earcut.triangulate_float64(verts, holes)

        t = [(int(idx[i]), int(idx[i+1]), int(idx[i+2])) for i in range(0, len(idx), 3)]
        v = [tuple(p) for p in verts]
        return t, v
    # Check earcut availability once
    earcut_available = True
    try:
        import mapbox_earcut  # noqa: F401
    except ImportError:
        earcut_available = False

    tris = None
    verts = None
    used_earcut = False

    if args.tri == 'earcut' and not earcut_available:
        raise SystemExit("mapbox_earcut not installed. pip install mapbox_earcut")
    print("\n--- DEBUG: Contour before triangulation ---")
    print(f"type(contour): {type(contour)}")
    print(f"len(contour): {len(contour)}")
    if isinstance(contour, np.ndarray):
        print(f"contour.shape: {contour.shape}")
        print("First 3 points:", contour[:3])
    elif isinstance(contour, (list, tuple)) and len(contour) > 0:
        print("First element type:", type(contour[0]))
        print("First 3 elements:", contour[:3])
    if args.tri in ('auto', 'earcut') and earcut_available:
        try:
            tris, verts = tri_earcut(contour)
            used_earcut = True
        except Exception as e:
            if args.tri == 'earcut':
                # show the true geometry error instead of a misleading install message
                raise SystemExit(f"earcut triangulation failed: {e}")
            # auto mode → fall back
            tris, verts = tri_earclip(contour)
    else:
        # earclip forced or earcut unavailable
        tris, verts = tri_earclip(contour)

    if not tris or len(tris) == 0:
        raise SystemExit(
            "Triangulation failed.\n"
            "Try: lower --eps (e.g., 0.4), increase --res/--aa, enable --close 1, "
            "and/or use --tri earcut."
        )

    # -------- Export --------
    obj_path = outdir / "vase2d.obj"
    export_obj(verts, tris, obj_path, scale=float(args.scale))
    print(f"Saved silhouette: {sil_path}")
    print(f"Saved 2D mesh OBJ: {obj_path} (V={len(verts)}, F={len(tris)}) [tri={('earcut' if used_earcut else 'earclip')}]")
    print("Done.")


if __name__ == '__main__':
    main()
