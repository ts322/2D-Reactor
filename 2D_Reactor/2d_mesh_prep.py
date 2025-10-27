#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLB/OBJ/PLY -> 2D silhouette.png (orthographic)

Dependencies: numpy, pillow, trimesh
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import trimesh

# -----------------------------
# Mesh load + normalize
# -----------------------------
def load_align_normalize(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path.as_posix(), force='mesh')
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(g for g in m.geometry.values()))
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
def silhouette_raster(
    mesh: trimesh.Trimesh,
    res=2048,
    margin=0.05,
    view='+y',
    aa=4,
    out_path: Path=None,
    rot_deg: float = 0.0,
    flipx: bool = False,
    flipy: bool = False,
):
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

    # In-plane rotation + optional flips BEFORE computing bounds
    if rot_deg:
        th = np.deg2rad(rot_deg)
        c, s = np.cos(th), np.sin(th)
        Uv, Vv = c*Uv - s*Vv, s*Uv + c*Vv
    if flipx:
        Uv = -Uv
    if flipy:
        Vv = -Vv

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
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='Input GLB/OBJ/PLY')
    ap.add_argument('--outdir', required=True, help='Output directory')
    ap.add_argument('--res', type=int, default=512, help='Silhouette resolution')
    ap.add_argument('--view', choices=['+y','+x','+z'], default='+y', help='Orthographic view')
    ap.add_argument('--margin', type=float, default=0.05, help='Silhouette frame margin')
    ap.add_argument('--aa', type=int, default=4, help='Antialiasing factor (supersampling)')
    ap.add_argument('--rot', type=float, default=0.0, help='In-plane rotation (degrees, +CCW)')
    ap.add_argument('--flipx', type=int, default=0, help='Flip U axis after rotation (0/1)')
    ap.add_argument('--flipy', type=int, default=0, help='Flip V axis after rotation (0/1)')
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    mesh = load_align_normalize(Path(args.inp))
    sil_path = outdir / "silhouette.png"
    silhouette_raster(
        mesh,
        res=args.res,
        margin=args.margin,
        view=args.view,
        aa=args.aa,
        out_path=sil_path,
        rot_deg=args.rot,
        flipx=bool(args.flipx),
        flipy=bool(args.flipy),
    )
    print(f"Saved silhouette: {sil_path}")

if __name__ == '__main__':
    main()
