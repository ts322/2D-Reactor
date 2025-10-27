#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG silhouette -> OpenFOAM mesh using your existing pipeline:
- reads white-on-black PNG
- OpenCV contour -> simplify -> resample -> (x, y_top, y_bot)
- optional area constraint (±10% default)
- optional rectangular outlet snap
- writes points into Example/system/blockMeshDict (polyLine edges)
- runs blockMesh + checkMesh via Docker

Requires: opencv-python, numpy, matplotlib (for preview)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import interpolate
import shutil
import os
import sys
from scipy.stats import qmc
from datetime import datetime
import math
import uuid
import subprocess
from typing import List, Tuple, Optional

# ----------------- knobs -----------------
THICKNESS = 0.1
MIN_GAP   = 0.00          # 0.02 if you want a guaranteed gap; 0 if not
PLOT_GEOMETRY = True
DELETE_ON_SEVERE = True
N_ADD = 50
BASE_N = 180

USE_IMAGE = True
IMG_PATH = "/Users/marcobarbacci/2D-Reactor/2D_Reactor/out/silhouette.png"
LENGTH_X = 3.0
SAMPLE_X = BASE_N + 2*N_ADD

# outlet controls
ENFORCE_OUTLET   = True
OUTLET_FRAC      = 0.03        # last 3% of domain set to a box
OUTLET_HEIGHT    = None        # set float to force, or None to auto from neighborhood
OUTLET_RAMP_PTS  = 6

# area constraint (area between y_top and y_bot along x, in case units)
TARGET_AREA = None             # e.g. 1.2; None to disable
AREA_TOL    = 0.10             # ±10%
AREA_MAX_IT = 3

ROT_DEG = 90.0       # +CCW; set 0.0 if you don’t want rotation
FLIP_X  = False      # mirror left↔right after rotation
FLIP_Y  = False  
# ----------------------------------------

def docker_run(cmd: List[str], case_path: str, image: str = "opencfd/openfoam-default:2506") -> subprocess.CompletedProcess:
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.path.abspath(case_path)}:/home/openfoam/case",
        "-w", "/home/openfoam/case", image
    ] + cmd
    return subprocess.run(docker_cmd, capture_output=True, text=True)

# ---------- helpers you already had ----------
def format_point(x, y, z):
    return f"({x:.8f} {y:.8f} {z:.8f})"

def dedupe_str_points(seq):
    out, last = [], None
    for s in seq:
        if s != last:
            out.append(s); last = s
    return out

def find_block(lines, key):
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith(key):
            start = i; break
    if start is None:
        raise RuntimeError(f"Couldn't find '{key}' block in blockMeshDict")
    has_inline_open = "(" in lines[start]
    open_idx = start if has_inline_open else start + 1
    end_idx = None
    for j in range(open_idx, len(lines)):
        if lines[j].strip() == ");":
            end_idx = j; break
    if end_idx is None:
        raise RuntimeError(f"Couldn't find end of '{key}' block (');')")
    return start, open_idx, end_idx

def rotate_image_keep_full(img: np.ndarray, deg: float) -> np.ndarray:
    if abs(deg) < 1e-9:
        return img
    h, w = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0, 2] += (nW / 2.0) - c[0]
    M[1, 2] += (nH / 2.0) - c[1]
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_NEAREST, borderValue=0)


def load_outer_contour(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    # --- NEW: rotate + optional flips BEFORE threshold/contours ---
    if abs(ROT_DEG) > 1e-9:
        img = rotate_image_keep_full(img, ROT_DEG)
    if FLIP_X:
        img = cv2.flip(img, 1)  # horizontal
    if FLIP_Y:
        img = cv2.flip(img, 0)  # vertical
    # --------------------------------------------------------------

    # White silhouette on black (invert if yours is black-on-white)
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No external contour found in image")

    c = max(contours, key=cv2.contourArea)[:, 0, :]  # (N,2)
    if c.shape[0] < 3:
        raise RuntimeError("Contour too small")

    # start near leftmost point (helps stability)
    i0 = int(np.argmin(c[:, 0]))
    c = np.vstack([c[i0:], c[:i0]]).astype(np.float32)
    return c


def simplify_polyline_rdp(poly_xy: np.ndarray, eps: float):
    p = poly_xy.astype(np.float32).reshape(-1, 1, 2)
    eps = float(max(eps, 1e-6))
    approx = cv2.approxPolyDP(p, epsilon=eps, closed=True)
    approx = approx[:, 0, :]
    if approx.shape[0] < 3:
        approx = poly_xy.astype(np.float32)
    return approx

def resample_closed_polyline(poly: np.ndarray, n_pts: int):
    p = poly
    if not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[0]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.hstack([[0.0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n_pts+1)[:-1]
    x = np.interp(t, s, p[:, 0]); y = np.interp(t, s, p[:, 1])
    return np.c_[x, y]

def ensure_strict_increasing(x):
    x = x.copy()
    for i in range(1, len(x)):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-9
    return x

def contour_to_channel_xy(poly_xy: np.ndarray, n_x: int = 400):
    poly = resample_closed_polyline(poly_xy, n_pts=2000)
    xmin, xmax = poly[:, 0].min(), poly[:, 0].max()
    x_all = np.linspace(xmin, xmax, n_x)
    buckets = [[] for _ in range(n_x)]
    idx = np.clip(np.floor((poly[:, 0] - xmin) / (xmax - xmin) * (n_x - 1)).astype(int), 0, n_x-1)
    for y, b in zip(poly[:, 1], idx):
        buckets[b].append(float(y))

    def nearest_nonempty(i):
        k = 1
        while True:
            l = i-k; r = i+k
            if l >= 0 and buckets[l]: return buckets[l]
            if r < n_x and buckets[r]: return buckets[r]
            k += 1

    y_top = np.empty_like(x_all); y_bot = np.empty_like(x_all)
    for i in range(n_x):
        ys = buckets[i] if buckets[i] else nearest_nonempty(i)
        ys = np.array(ys)
        y_bot[i] = ys.max()
        y_top[i] = ys.min()

    # flip vertical image coords → Cartesian
    ymax = poly[:, 1].max()
    y_top = ymax - y_top
    y_bot = ymax - y_bot
    x_all = ensure_strict_increasing(x_all)
    return x_all, y_top, y_bot

def scale_to_length_and_lift(x, y1, y2, Lx_target=3.0):
    xmin, xmax = x.min(), x.max()
    sx = Lx_target / (xmax - xmin)
    x_  = (x - xmin) * sx
    y2min = y2.min()
    y1_ = (y1 - y2min) * sx
    y2_ = (y2 - y2min) * sx
    return x_, y1_, y2_

def area_between_curves(x, y_top, y_bot):
    return float(np.trapz((y_top - y_bot), x))

def enforce_area(x, y_top, y_bot, target_area: float, tol: float = 0.10, max_iter: int = 3):
    """Uniform isotropic scaling to match target area (area ∝ s^2)."""
    if target_area is None:
        return x, y_top, y_bot
    A = area_between_curves(x, y_top, y_bot)
    if A <= 0:
        return x, y_top, y_bot
    s = 1.0
    for _ in range(max(1, int(max_iter))):
        s *= np.sqrt(max(1e-16, target_area / A))
        x_s  = x * s
        yT_s = y_top * s
        yB_s = y_bot * s
        # keep left at x=0 (optional): shift back so min(x)=0 to preserve origin
        x_s -= x_s.min()
        A = area_between_curves(x_s, yT_s, yB_s)
        if abs(A - target_area) <= tol * target_area:
            return x_s, yT_s, yB_s
    return x_s, yT_s, yB_s

def enforce_outlet_box(x, y_top, y_bot,
                       L_frac=0.03, height=None, align='auto', ramp_pts=6):
    """Make last L_frac of domain a rectangular outlet of constant height."""
    x = x.copy(); yT = y_top.copy(); yB = y_bot.copy()
    Lx = x.max() - x.min()
    i0 = np.searchsorted(x, x.max() - L_frac * Lx)
    i0 = max(2, min(i0, len(x) - 2))
    j0 = max(0, i0 - max(ramp_pts, 3))
    mid_loc = 0.5 * np.median(yT[j0:i0] + yB[j0:i0]) if i0 > j0 else 0.5 * (yT[i0-1] + yB[i0-1])
    H_loc   = np.median(yT[j0:i0] - yB[j0:i0])       if i0 > j0 else (yT[i0-1] - yB[i0-1])
    H = H_loc if (height is None) else float(height)

    if align == 'auto':
        yB_tgt = mid_loc - 0.5 * H; yT_tgt = mid_loc + 0.5 * H
    elif align == 'bottom':
        yB_tgt = float(np.median(yB[j0:i0])); yT_tgt = yB_tgt + H
    elif align == 'top':
        yT_tgt = float(np.median(yT[j0:i0])); yB_tgt = yT_tgt - H
    else:
        raise ValueError("align must be 'auto'|'bottom'|'top'")

    k0 = max(0, i0 - ramp_pts)
    if i0 > k0:
        yB[k0:i0] = np.linspace(yB[k0], yB_tgt, i0 - k0)
        yT[k0:i0] = np.linspace(yT[k0], yT_tgt, i0 - k0)
    yB[i0:] = yB_tgt; yT[i0:] = yT_tgt

    # pin last x to xmax and ensure strict monotonicity
    xmax = x.max()
    x[-1] = xmax
    if x[-2] >= xmax: x[-2] = xmax - 1e-9
    x = ensure_strict_increasing(x)
    return x, yT, yB

def image_to_channel_curves(
    img_path: str,
    rdp_eps_px: float = 1.5,
    n_resample_perimeter: int = 3000,
    n_x: int = SAMPLE_X,
    Lx_target: float = LENGTH_X,
    target_area: Optional[float] = TARGET_AREA,
    area_tol: float = AREA_TOL,
    area_max_iter: int = AREA_MAX_IT,
    enforce_outlet: bool = ENFORCE_OUTLET,
    outlet_frac: float = OUTLET_FRAC,
    outlet_height: Optional[float] = OUTLET_HEIGHT,
    outlet_ramp_pts: int = OUTLET_RAMP_PTS,
):
    c = load_outer_contour(img_path)
    c = simplify_polyline_rdp(c, eps=rdp_eps_px)
    c = resample_closed_polyline(c, n_resample_perimeter)
    x_all, y_top, y_bot = contour_to_channel_xy(c, n_x=n_x)
    x_all, y_top, y_bot = scale_to_length_and_lift(x_all, y_top, y_bot, Lx_target=Lx_target)

    # area constraint (uniform scaling)
    if target_area is not None:
        x_all, y_top, y_bot = enforce_area(x_all, y_top, y_bot, target_area, area_tol, area_max_iter)

    # outlet rectangular box (optional)
    if enforce_outlet and outlet_frac > 0:
        x_all, y_top, y_bot = enforce_outlet_box(
            x_all, y_top, y_bot,
            L_frac=outlet_frac, height=outlet_height, align='auto', ramp_pts=outlet_ramp_pts
        )

    # final safety
    if MIN_GAP > 0:
        y_top = np.maximum(y_top, y_bot + MIN_GAP)

    # format for blockMesh
    l11 = [format_point(x_all[i], y_top[i], 0.0)          for i in range(len(x_all))]
    l12 = [format_point(x_all[i], y_top[i], THICKNESS)    for i in range(len(x_all))]
    l21 = [format_point(x_all[i], y_bot[i], 0.0)          for i in range(len(x_all))]
    l22 = [format_point(x_all[i], y_bot[i], THICKNESS)    for i in range(len(x_all))]
    l11 = dedupe_str_points(l11); l12 = dedupe_str_points(l12)
    l21 = dedupe_str_points(l21); l22 = dedupe_str_points(l22)

    return l11, l12, l21, l22, x_all, y_top, y_bot

# ---------- your writer (unchanged API) ----------
def write_vertices_and_edges(path, x_all, y_top, y_bot, l11, l12, l21, l22):
    dict_path = os.path.join(path, "system", "blockMeshDict")
    with open(dict_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    xmin = float(x_all[0]);  xmax = float(x_all[-1])
    ybot_L = float(y_bot[0]); ybot_R = float(y_bot[-1])
    ytop_L = float(y_top[0]); ytop_R = float(y_top[-1])

    vertices_block = [
        "vertices", "(",
        f"    {format_point(xmin, ybot_L, 0.0)}",       # 0
        f"    {format_point(xmax, ybot_R, 0.0)}",       # 1
        f"    {format_point(xmax, ytop_R, 0.0)}",       # 2
        f"    {format_point(xmin, ytop_L, 0.0)}",       # 3
        f"    {format_point(xmin, ybot_L, THICKNESS)}", # 4
        f"    {format_point(xmax, ybot_R, THICKNESS)}", # 5
        f"    {format_point(xmax, ytop_R, THICKNESS)}", # 6
        f"    {format_point(xmin, ytop_L, THICKNESS)}", # 7
        ");"
    ]
    v_start, _, v_end = find_block(lines, "vertices")
    lines = lines[:v_start] + vertices_block + lines[v_end+1:]

    edges_block = [
        "edges", "(",
        "\tpolyLine 0 1", "\t("] + [ "\t\t"+pt for pt in l21 ] + ["\t)",
        "\tpolyLine 4 5", "\t("] + [ "\t\t"+pt for pt in l22 ] + ["\t)",
        "\tpolyLine 3 2", "\t("] + [ "\t\t"+pt for pt in l11 ] + ["\t)",
        "\tpolyLine 7 6", "\t("] + [ "\t\t"+pt for pt in l12 ] + ["\t)",
        ");"
    ]
    e_start, _, e_end = find_block(lines, "edges")
    lines = lines[:e_start] + edges_block + lines[e_end+1:]

    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# ---------- build ----------
def build_mesh(path):
    shutil.copytree("Example", path, dirs_exist_ok=True)

    l11, l12, l21, l22, x_all, y_top, y_bot = image_to_channel_curves(
        IMG_PATH,
        rdp_eps_px=1.5,
        n_resample_perimeter=3000,
        n_x=SAMPLE_X,
        Lx_target=LENGTH_X,
        target_area=TARGET_AREA,
        area_tol=AREA_TOL,
        area_max_iter=AREA_MAX_IT,
        enforce_outlet=ENFORCE_OUTLET,
        outlet_frac=OUTLET_FRAC,
        outlet_height=OUTLET_HEIGHT,
        outlet_ramp_pts=OUTLET_RAMP_PTS,
    )

    if PLOT_GEOMETRY:
        plt.figure()
        plt.plot(x_all, y_top, label="top")
        plt.plot(x_all, y_bot, label="bot")
        plt.plot([x_all[0], x_all[0]], [y_bot[0], y_top[0]], 'g-')   # left wall
        plt.plot([x_all[-1], x_all[-1]], [y_bot[-1], y_top[-1]], 'r-')  # right wall
        plt.grid(); plt.legend()
        plt.title("Reactor Geometry from PNG")
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "reactor_geometry.png"))
        plt.close()

    write_vertices_and_edges(path, x_all, y_top, y_bot, l11, l12, l21, l22)

    # run OpenFOAM
    bmesh = docker_run(["blockMesh"], path, image="opencfd/openfoam-default:2506")
    print(bmesh.stdout)
    cmesh = docker_run(["checkMesh"], path, image="opencfd/openfoam-default:2506")
    out = cmesh.stdout
    print(out)

    # parse & optionally delete on severe issues
    mesh_size = None
    for line in out.splitlines():
        if line.strip().startswith("cells:"):
            try: mesh_size = int(line.split()[-1])
            except: pass
            break

    severe_patterns = [
        "Zero or negative face area",
        "Zero or negative cell volume",
        "incorrectly oriented",
    ]
    severe_hits = [p for p in severe_patterns if p in out]

    if severe_hits and DELETE_ON_SEVERE:
        print(f"[SEVERE] Deleting {path} due to:")
        for p in severe_hits: print(f"   - {p}")
        for line in out.splitlines():
            if any(key in line for key in severe_patterns) or "Failed" in line:
                print("   " + line)
        shutil.rmtree(path)
        return None
    else:
        if "Failed" in out:
            print(f"[WARN] Mesh quality issues for {path}:")
            for line in out.splitlines():
                if line.strip().startswith("*") or "Failed" in line:
                    print("   " + line.strip())
    return mesh_size

# ---------- main sweep (kept like yours) ----------
if __name__ == "__main__":
    mesh_sizes = []
    num_samples = 1   # set to 1 since image path is fixed; bump if you want repeats
    for i in range(num_samples):
        ID = f"Geometry_{i+1}"
        path = os.path.join(os.path.expanduser("~/2D-Reactor/2D_Reactor/Mesh"), ID)
        try:
            mesh_size = build_mesh(path)
        except Exception as e:
            print(f"[ERROR] Build failed for {path}: {e}")
            mesh_size = None
            if os.path.isdir(path):
                shutil.rmtree(path)
        mesh_sizes.append(mesh_size)
    print(mesh_sizes)
