# Requires: pip install opencv-python shapely (shapely optional but handy)
import cv2
import numpy as np
from ntpath import join
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import interpolate
import shutil
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from scipy.stats import qmc
from datetime import datetime
import math
import uuid
import subprocess
from typing import List, Tuple, Optional

THICKNESS = 0.1                 
TARGET_END_WIDTH = 0.5         
MIN_GAP = 0.02                  
PLOT_GEOMETRY = True
DELETE_ON_SEVERE = True         
N_ADD = 50                      
BASE_N = 180    
USE_IMAGE = True
IMG_PATH = "/Users/marcobarbacci/2D-Reactor/2D_Reactor/out/silhouette.png"   # your vase png
LENGTH_X = 3.0                          # case units in x
SAMPLE_X = BASE_N + 2*N_ADD             # similar density as before  

def load_outer_contour(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    # For white-on-black shapes (your PNG). Invert this if your colors are reversed.
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # OpenCV 4 returns (contours, hierarchy)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No external contour found in image")

    # Largest contour
    c = max(contours, key=cv2.contourArea)   # (N,1,2)
    c = c[:, 0, :]                           # -> (N,2)
    if c.shape[0] < 3:
        raise RuntimeError("Contour too small")

    # start near the leftmost point (optional)
    i0 = int(np.argmin(c[:, 0]))
    c = np.vstack([c[i0:], c[:i0]])

    # IMPORTANT: cast to float32 for approxPolyDP
    return c.astype(np.float32)

def simplify_polyline_rdp(poly_xy: np.ndarray, eps: float):
    """
    Douglas–Peucker simplification. Requires float32 or int32.
    """
    if poly_xy is None or len(poly_xy) == 0:
        raise ValueError("simplify_polyline_rdp: empty polyline")

    p = poly_xy.astype(np.float32).reshape(-1, 1, 2)
    if eps <= 0:
        eps = 1.0  # fallback

    approx = cv2.approxPolyDP(p, epsilon=float(eps), closed=True)
    approx = approx[:, 0, :]
    if approx.shape[0] < 3:
        # if RDP was too aggressive, fall back to the original
        approx = poly_xy.astype(np.float32)
    return approx

def resample_closed_polyline(poly: np.ndarray, n_pts: int):
    """Uniform arc-length resample a closed polyline to n_pts (2D array)."""
    # ensure closed
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.hstack([[0.0], np.cumsum(seg)])        # cumulative length
    L = s[-1]
    t = np.linspace(0, L, n_pts+1)[:-1]           # drop duplicated endpoint
    x = np.interp(t, s, poly[:,0])
    y = np.interp(t, s, poly[:,1])
    return np.c_[x, y]

def contour_to_channel_xy(poly_xy: np.ndarray, n_x: int = 400):
    """
    Convert a closed contour into channel-style arrays:
    x_all (ascending), y_top(x), y_bot(x).

    Assumes that for each x there are exactly two boundary points
    (top and bottom). Your vase satisfies this (no overhangs).
    """
    # Work in a normalized frame
    xmin, xmax = poly_xy[:,0].min(), poly_xy[:,0].max()
    ymin, ymax = poly_xy[:,1].min(), poly_xy[:,1].max()

    # x bins
    x_all = np.linspace(xmin, xmax, n_x)
    y_top = np.empty_like(x_all)
    y_bot = np.empty_like(x_all)

    # For speed, bucket points by nearest x bin
    bin_idx = np.clip(
        np.floor((poly_xy[:,0] - xmin) / (xmax - xmin) * (n_x - 1)).astype(int),
        0, n_x-1
    )

    # collect y's per bin
    buckets = [[] for _ in range(n_x)]
    for (y, b) in zip(poly_xy[:,1], bin_idx):
        buckets[b].append(float(y))

    # some bins may be empty (use neighborhood fill)
    def nearest_nonempty(i):
        L = len(buckets)
        k = 1
        while True:
            l = i - k; r = i + k
            if l >= 0 and buckets[l]: return buckets[l]
            if r < L and buckets[r]: return buckets[r]
            k += 1

    for i in range(n_x):
        ys = buckets[i] if buckets[i] else nearest_nonempty(i)
        ys = np.array(ys)
        y_top[i] = ys.min()   # note: image y grows downward
        y_bot[i] = ys.max()

    # convert image coordinates (origin top-left) to Cartesian (origin bottom-left)
    # Flip vertically: Yc = ymax - y_img
    y_top = ymax - y_top
    y_bot = ymax - y_bot
    x_all = x_all.copy()

    # Make x strictly increasing and unique
    dx = np.diff(x_all)
    if np.any(dx <= 0):
        for i in range(1, len(x_all)):
            if x_all[i] <= x_all[i-1]:
                x_all[i] = x_all[i-1] + 1e-9

    return x_all, y_top, y_bot

def scale_and_shift(x, y1, y2, Lx_target=3.0, y0_shift_to_zero=True):
    """
    Map pixel units to physical units preserving aspect ratio.
    Lx_target = desired domain length in your case coords.
    """
    xmin, xmax = x.min(), x.max()
    sx = Lx_target / (xmax - xmin)
    x_ = (x - xmin) * sx
    y1_ = (y1 - y1.min()) * sx if y0_shift_to_zero else y1 * sx
    y2_ = (y2 - y1.min()) * sx if y0_shift_to_zero else y2 * sx
    return x_, y1_, y2_

def image_to_channel_curves(
    img_path: str,
    rdp_eps_px: float = 2.0,
    n_resample_perimeter: int = 2000,
    n_x: int = 400,
    Lx_target: float = 3.0
):
    """Full pipeline: image -> outer contour -> simplified -> resampled -> (x, y_top, y_bot) in case units."""
    c = load_outer_contour(img_path)
    c = simplify_polyline_rdp(c, eps=rdp_eps_px)
    c = resample_closed_polyline(c, n_resample_perimeter)
    x_all, y_top, y_bot = contour_to_channel_xy(c, n_x=n_x)
    x_all, y_top, y_bot = scale_and_shift(x_all, y_top, y_bot, Lx_target=Lx_target)
    return x_all, y_top, y_bot


# --- helpers you already have somewhere ---
# - image_to_channel_curves(img_path, rdp_eps_px, n_resample_perimeter, n_x, Lx_target)
# - format_point, dedupe_str_points
# - MIN_GAP, THICKNESS, BASE_N, N_ADD

def enforce_outlet(x, y_top, y_bot, L_frac=0.04, height=None, blend_pts=40, align='auto'):
    x = x.copy(); yT = y_top.copy(); yB = y_bot.copy()
    Lx = x.max() - x.min()
    i0 = np.searchsorted(x, x.max() - L_frac * Lx)
    j0 = max(0, i0 - max(blend_pts, 10))
    yB_ref = float(np.median(yB[j0:i0])) if i0 > j0 else yB[i0-1]
    h_ref  = float(np.median(yT[j0:i0] - yB[j0:i0])) if i0 > j0 else (yT[i0-1]-yB[i0-1])
    H = h_ref if height is None else float(height)

    if align == 'auto':
        mid = float(np.median(0.5*(yT[j0:i0] + yB[j0:i0]))) if i0 > j0 else 0.5*(yT[i0-1]+yB[i0-1])
        yB_tgt, yT_tgt = mid - 0.5*H, mid + 0.5*H
    elif align == 'bottom':
        yB_tgt, yT_tgt = yB_ref, yB_ref + H
    elif align == 'top':
        yTref = float(np.median(yT[j0:i0])) if i0 > j0 else yT[i0-1]
        yB_tgt, yT_tgt = yTref - H, yTref
    else:
        raise ValueError("align must be 'auto'|'bottom'|'top'")

    # hard set outlet
    yB[i0:] = yB_tgt; yT[i0:] = yT_tgt
    # linear blend into it
    k0 = max(0, i0 - blend_pts)
    if i0 > k0:
        yB[k0:i0] = np.linspace(yB[k0], yB_tgt, i0-k0)
        yT[k0:i0] = np.linspace(yT[k0], yT_tgt, i0-k0)

    # keep x strictly increasing
    for i in range(1, len(x)):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-9
    return x, yT, yB


def build_arrays_from_image_safe(
    img_path,
    rdp_eps_px=2.0,
    n_resample_perimeter=2000,
    n_x=None,
    Lx_target=3.0,
    outlet_height=None,          # set e.g. 0.20 or leave None to auto
    outlet_frac=0.04,
    outlet_blend_pts=40,
):
    """Returns l11,l12,l21,l22,x_all,y_top,y_bot or raises RuntimeError with details."""
    if n_x is None:
        n_x = BASE_N + 2*N_ADD

    # initialize to avoid UnboundLocalError in case of early exception
    x_all = y_top = y_bot = None

    try:
        # 1) image -> curves
        x_all, y_top, y_bot = image_to_channel_curves(
            img_path,
            rdp_eps_px=rdp_eps_px,
            n_resample_perimeter=n_resample_perimeter,
            n_x=n_x,
            Lx_target=Lx_target,
        )

        if x_all is None or len(x_all) == 0:
            raise RuntimeError("image_to_channel_curves returned empty arrays")

        # 2) enforce outlet (optional but recommended)
        x_all, y_top, y_bot = enforce_outlet(
            x_all, y_top, y_bot,
            L_frac=outlet_frac,
            height=outlet_height,
            blend_pts=outlet_blend_pts,
            align='auto'
        )

        # 3) safety gap if you still want it
        y_top = np.maximum(y_top, y_bot + MIN_GAP)

        # 4) format for blockMesh
        l11 = [format_point(x_all[i], y_top[i], 0.0)          for i in range(len(x_all))]
        l12 = [format_point(x_all[i], y_top[i], THICKNESS)    for i in range(len(x_all))]
        l21 = [format_point(x_all[i], y_bot[i], 0.0)          for i in range(len(x_all))]
        l22 = [format_point(x_all[i], y_bot[i], THICKNESS)    for i in range(len(x_all))]

        l11 = dedupe_str_points(l11); l12 = dedupe_str_points(l12)
        l21 = dedupe_str_points(l21); l22 = dedupe_str_points(l22)

        return l11, l12, l21, l22, x_all, y_top, y_bot

    except Exception as e:
        # Make the failure explicit and upstream-safe
        raise RuntimeError(f"Image-based geometry failed: {e}") from e


def docker_run(cmd: List[str], case_path: str, image: str = "opencfd/openfoam-default:2506") -> subprocess.CompletedProcess:
    """Run an OpenFOAM command inside a Docker container."""
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.path.abspath(case_path)}:/home/openfoam/case",
        "-w", "/home/openfoam/case", image
    ] + cmd
    return subprocess.run(docker_cmd, capture_output=True, text=True)   

def sin_line(x, b, a, c, off):
    return b + (a * np.sin(c * (x - off)))

def smooth_local(x, y, idx, k=5):
    i0 = max(idx - k, 0)
    i2 = min(idx + k, len(x)-1)
    xm = [x[i0], x[idx], x[i2]]
    ym = [y[i0], ((y[i0] + y[i2]) * 0.5 + y[idx]) * 0.5, y[i2]]
    x_new = np.linspace(x[i0], x[i2], (i2 - i0) + 1)
    y_new = interp1d(xm, ym, kind='quadratic')(x_new)

    x[i0:i2+1] = x_new
    y[i0:i2+1] = y_new
    return x, y

def format_point(x, y, z):
    return f"({x:.8f} {y:.8f} {z:.8f})"

def dedupe_str_points(seq):
    out = []
    last = None
    for s in seq:
        if s != last:
            out.append(s)
            last = s
    return out

def find_block(lines, key):
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith(key):
            start = i
            break
    if start is None:
        raise RuntimeError(f"Couldn't find '{key}' block in blockMeshDict")

    has_inline_open = "(" in lines[start]
    open_idx = start if has_inline_open else start + 1

    end_idx = None
    for j in range(open_idx, len(lines)):
        if lines[j].strip() == ");":
            end_idx = j
            break
    if end_idx is None:
        raise RuntimeError(f"Couldn't find end of '{key}' block (');')")
    return start, open_idx, end_idx

def build_arrays(p1, p2, p3):
    if not (0.1 <= p1 <= 0.5):
        raise ValueError(f"p1 out of range [0.1, 0.5]: {p1}")
    if not (3.0 <= p2 <= 6.0):
        raise ValueError(f"p2 out of range [3, 6]: {p2}")
    if not (0.0 <= p3 <= np.pi/2):
        raise ValueError(f"p3 out of range [0, pi/2]: {p3}")

    x_core = np.linspace(0.0, 3.0, BASE_N)


    y_top_core = sin_line(x_core, 0.5, p1, p2, p3)    
    y_bot_core = sin_line(x_core, 0.0, 0.25, p2, p3)   

    add_start_x = np.linspace(x_core[0] - 1.0, x_core[0], N_ADD + 1, endpoint=True)[:-1]
    add_end_x   = np.linspace(x_core[-1], x_core[-1] + 1.0, N_ADD + 1, endpoint=True)[1:]

    y_top_start_L = y_bot_core[0] + TARGET_END_WIDTH
    y_top_end_R   = y_bot_core[-1] + TARGET_END_WIDTH

    f_y1_start = interpolate.interp1d(
        [add_start_x[0], add_start_x[-1]],
        [y_top_start_L,  y_top_core[0]],
        kind='linear'
    )
    f_y1_end = interpolate.interp1d(
        [add_end_x[0], add_end_x[-1]],
        [y_top_core[-1], y_top_end_R],
        kind='linear'
    )

    y_top = np.concatenate([f_y1_start(add_start_x), y_top_core, f_y1_end(add_end_x)])
    y_bot = np.concatenate([np.full_like(add_start_x, y_bot_core[0]), y_bot_core, np.full_like(add_end_x, y_bot_core[-1])])
    x_all = np.concatenate([add_start_x, x_core, add_end_x])

    x_all, y_top = smooth_local(x_all, y_top, idx=N_ADD) 
    x_all, y_bot = smooth_local(x_all, y_bot, idx=N_ADD)
    x_all, y_top = smooth_local(x_all, y_top, idx=N_ADD + BASE_N)
    x_all, y_bot = smooth_local(x_all, y_bot, idx=N_ADD + BASE_N)

    y_top = np.maximum(y_top, y_bot + MIN_GAP)

    if not np.all(np.diff(x_all) > 0):
        order = np.argsort(x_all)
        x_all = x_all[order]
        y_top = y_top[order]
        y_bot = y_bot[order]
        dx = np.diff(x_all)
        if np.any(dx <= 0):
            eps = 1e-9
            for i in range(1, len(x_all)):
                if x_all[i] <= x_all[i-1]:
                    x_all[i] = x_all[i-1] + eps

    l11 = [format_point(x_all[i], y_top[i], 0.0)     for i in range(len(x_all))]
    l12 = [format_point(x_all[i], y_top[i], THICKNESS) for i in range(len(x_all))]
    l21 = [format_point(x_all[i], y_bot[i], 0.0)     for i in range(len(x_all))]
    l22 = [format_point(x_all[i], y_bot[i], THICKNESS) for i in range(len(x_all))]

    l11 = dedupe_str_points(l11)
    l12 = dedupe_str_points(l12)
    l21 = dedupe_str_points(l21)
    l22 = dedupe_str_points(l22)

    return l11, l12, l21, l22, x_all, y_top, y_bot

def write_vertices_and_edges(path, x_all, y_top, y_bot, l11, l12, l21, l22):
    dict_path = os.path.join(path, "system", "blockMeshDict")
    with open(dict_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    v0 = format_point(x_all[0],  y_bot[0 if len(y_bot)==0 else 0].split()[1][1:] if False else y_bot[0 if True else 0], 0.0)

    xmin = float(x_all[0])
    xmax = float(x_all[-1])
    ybot_L = float(y_bot[0])
    ybot_R = float(y_bot[-1])
    ytop_L = float(y_top[0])
    ytop_R = float(y_top[-1])

    vertices_block = []
    vertices_block.append("vertices")
    vertices_block.append("(")
    vertices_block.append(f"    {format_point(xmin, ybot_L, 0.0)}")       # 0
    vertices_block.append(f"    {format_point(xmax, ybot_R, 0.0)}")       # 1
    vertices_block.append(f"    {format_point(xmax, ytop_R, 0.0)}")       # 2
    vertices_block.append(f"    {format_point(xmin, ytop_L, 0.0)}")       # 3
    vertices_block.append(f"    {format_point(xmin, ybot_L, THICKNESS)}") # 4
    vertices_block.append(f"    {format_point(xmax, ybot_R, THICKNESS)}") # 5
    vertices_block.append(f"    {format_point(xmax, ytop_R, THICKNESS)}") # 6
    vertices_block.append(f"    {format_point(xmin, ytop_L, THICKNESS)}") # 7
    vertices_block.append(");")

    v_start, v_open, v_end = find_block(lines, "vertices")
    lines = lines[:v_start] + vertices_block + lines[v_end+1:]

    edges_block = []
    edges_block.append("edges")
    edges_block.append("(")
    edges_block.append("\tpolyLine 0 1")
    edges_block.append("\t(")
    for pt in l21:
        edges_block.append("\t\t" + pt)
    edges_block.append("\t)")
    edges_block.append("\tpolyLine 4 5")
    edges_block.append("\t(")
    for pt in l22:
        edges_block.append("\t\t" + pt)
    edges_block.append("\t)")
    edges_block.append("\tpolyLine 3 2")
    edges_block.append("\t(")
    for pt in l11:
        edges_block.append("\t\t" + pt)
    edges_block.append("\t)")
    edges_block.append("\tpolyLine 7 6")
    edges_block.append("\t(")
    for pt in l12:
        edges_block.append("\t\t" + pt)
    edges_block.append("\t)")
    edges_block.append(");")

    e_start, e_open, e_end = find_block(lines, "edges")
    lines = lines[:e_start] + edges_block + lines[e_end+1:]

    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_mesh(p1, p2, p3, path):
    shutil.copytree("Example", path, dirs_exist_ok=True)

    try:
        l11, l12, l21, l22, x_all, y_top, y_bot = build_arrays_from_image_safe(
            IMG_PATH,
            rdp_eps_px=1.5,                # start conservative
            n_resample_perimeter=3000,
            n_x=BASE_N + 2*N_ADD,
            Lx_target=3.0,
            outlet_height=None,            # or None to auto
            outlet_frac=0.04,
            outlet_blend_pts=40,
        )
    except Exception as e:
        # bubble up so your outer try/except prints the reason and cleans the case
        raise

    if PLOT_GEOMETRY:
        plt.figure()
        plt.plot(x_all, y_top)
        plt.plot(x_all, y_bot)
        # draw the verticals so it looks “closed”
        plt.plot([x_all[0], x_all[0]], [y_bot[0], y_top[0]])
        plt.plot([x_all[-1], x_all[-1]], [y_bot[-1], y_top[-1]])
        plt.xlim(min(x_all)-0.1, max(x_all)+0.1)
        plt.ylim(min(y_bot)-0.2, max(y_top)+0.2)
        plt.grid()
        plt.title(f"Reactor Geometry - p1={p1:.3f}, p2={p2:.3f}, p3={p3:.3f}")
        plt.savefig(os.path.join(path, "reactor_geometry.png"))
        plt.close()

    write_vertices_and_edges(path, x_all, y_top, y_bot, l11, l12, l21, l22)

    bmesh = docker_run(["blockMesh"], path, image="opencfd/openfoam-default:2506")
    cmesh = docker_run(["checkMesh"], path, image="opencfd/openfoam-default:2506")


    out = cmesh.stdout

    mesh_size = None
    for line in out.splitlines():
        if line.strip().startswith("cells:"):
            try:
                mesh_size = int(line.split()[-1])
            except Exception:
                pass
            break


    severe_patterns = [
        "Zero or negative face area",
        "Zero or negative cell volume",
        "incorrectly oriented",
    ]
    severe_hits = [p for p in severe_patterns if p in out]

    if severe_hits and DELETE_ON_SEVERE:
        print(f"[SEVERE] Deleting {path} due to:")
        for p in severe_hits:
            print(f"   - {p}")
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

    

if __name__ == "__main__":
    mesh_sizes = []

    num_samples = 10
    num_dimensions = 3

    sampler = qmc.LatinHypercube(d=num_dimensions)
    sample = sampler.random(n=num_samples)

    lower_bounds = [0.1, 3.0, 0.0]
    upper_bounds = [0.5, 6.0, math.pi/2]

    scaled_sample = qmc.scale(sample, lower_bounds, upper_bounds)

    for i in range(num_samples):
        p1, p2, p3 = map(float, scaled_sample[i])
        #ID = "{}{}".format(datetime.now().strftime('%Y%m_%d_%H_%M_%S'), uuid.uuid4().hex)
        ID = f"Geometry_{i+1}"
        path = os.path.join(os.path.expanduser("~/2D-Reactor/2D_Reactor/Mesh"), ID)
        try:
            mesh_size = build_mesh(p1, p2, p3, path)
        except Exception as e:
            print(f"[ERROR] Build failed for {path}: {e}")
            mesh_size = None
            if os.path.isdir(path):
                shutil.rmtree(path)
        mesh_sizes.append(mesh_size)

    print(mesh_sizes)