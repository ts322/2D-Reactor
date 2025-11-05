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
import argparse
import csv

# -------------------------
# Tunables / Project params
# -------------------------
THICKNESS = 0.1                 # z-thickness of the channel
TARGET_END_WIDTH = 0.5          # enforce inlet/outlet width (top - bottom) at both ends
MIN_GAP = 0.02                  # minimal gap between top and bottom everywhere (safety)
PLOT_GEOMETRY = True
DELETE_ON_SEVERE = True         # delete only on severe issues (zero/neg volume/area, bad orientation)
N_ADD = 50                      # number of points added at each end
BASE_N = 180                    # base number of points along the core section

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def sin_line(x, b, a, c, off):
    return b + (a * np.sin(c * (x - off)))

def smooth_local(x, y, idx, k=5):
    """
    Smooth a small neighborhood around index 'idx' using quadratic interpolation.
    Keeps x monotonic and avoids ringing.
    """
    i0 = max(idx - k, 0)
    i2 = min(idx + k, len(x)-1)
    xm = [x[i0], x[idx], x[i2]]
    ym = [y[i0], ((y[i0] + y[i2]) * 0.5 + y[idx]) * 0.5, y[i2]]
    x_new = np.linspace(x[i0], x[i2], (i2 - i0) + 1)
    y_new = interp1d(xm, ym, kind='quadratic')(x_new)

    # splice
    x[i0:i2+1] = x_new
    y[i0:i2+1] = y_new
    return x, y

def format_point(x, y, z):
    # keep numbers readable and stable for OpenFOAM
    return f"({x:.8f} {y:.8f} {z:.8f})"

def dedupe_str_points(seq):
    """Remove consecutive duplicate string points."""
    out = []
    last = None
    for s in seq:
        if s != last:
            out.append(s)
            last = s
    return out

def find_block(lines, key):
    """
    Find an OpenFOAM dictionary block by header 'key' and return (start_idx, open_idx, end_idx).
    Expects structure:
      key
      (
        ...
      );
    or: key ( ... );
    """
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith(key):
            start = i
            break
    if start is None:
        raise RuntimeError(f"Couldn't find '{key}' block in blockMeshDict")

    # Determine where '(' is
    has_inline_open = "(" in lines[start]
    open_idx = start if has_inline_open else start + 1
    # Find corresponding ');'
    end_idx = None
    for j in range(open_idx, len(lines)):
        if lines[j].strip() == ");":
            end_idx = j
            break
    if end_idx is None:
        raise RuntimeError(f"Couldn't find end of '{key}' block (');')")
    return start, open_idx, end_idx

# ------------------------------------------------
# Geometry builder
# ------------------------------------------------
def build_arrays(p1, p2, p3):
    """
    Return:
      x_line, y_top, y_bot, and polyline point strings for OpenFOAM:
      l11: top, z=0     (edge 3 2)
      l12: top, z=THK   (edge 7 6)
      l21: bottom, z=0  (edge 0 1)
      l22: bottom, z=THK (edge 4 5)
    All arrays are strictly increasing in x and the top is always above bottom.
    """
    # basic validity
    if not (0.1 <= p1 <= 0.5):
        raise ValueError(f"p1 out of range [0.1, 0.5]: {p1}")
    if not (3.0 <= p2 <= 6.0):
        raise ValueError(f"p2 out of range [3, 6]: {p2}")
    if not (0.0 <= p3 <= np.pi/2):
        raise ValueError(f"p3 out of range [0, pi/2]: {p3}")

    # Base x in [0, 3]
    x_core = np.linspace(0.0, 3.0, BASE_N)

    # Sine walls
    y_top_core = sin_line(x_core, 0.5, p1, p2, p3)     # top
    y_bot_core = sin_line(x_core, 0.0, 0.25, p2, p3)   # bottom

    # End extensions: strictly increasing!
    add_start_x = np.linspace(x_core[0] - 1.0, x_core[0], N_ADD + 1, endpoint=True)[:-1]
    add_end_x   = np.linspace(x_core[-1], x_core[-1] + 1.0, N_ADD + 1, endpoint=True)[1:]

    # Build start/end y for top to enforce consistent inlet/outlet width
    y_top_start_L = y_bot_core[0] + TARGET_END_WIDTH
    y_top_end_R   = y_bot_core[-1] + TARGET_END_WIDTH

    # Linear transitions for top only (bottom held constant on the pads)
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

    # Smooth a little around the two junctions to reduce kinks
    x_all, y_top = smooth_local(x_all, y_top, idx=N_ADD)           # left junction
    x_all, y_bot = smooth_local(x_all, y_bot, idx=N_ADD)
    x_all, y_top = smooth_local(x_all, y_top, idx=N_ADD + BASE_N)  # right junction
    x_all, y_bot = smooth_local(x_all, y_bot, idx=N_ADD + BASE_N)

    # Safety: enforce minimal gap everywhere to avoid collapse
    y_top = np.maximum(y_top, y_bot + MIN_GAP)

    # Final sanity: x must be strictly increasing
    if not np.all(np.diff(x_all) > 0):
        # As a last resort, sort by x and reorder top/bottom to match
        order = np.argsort(x_all)
        x_all = x_all[order]
        y_top = y_top[order]
        y_bot = y_bot[order]
        # and ensure strict monotonicity by nudging duplicates (shouldn't normally happen)
        dx = np.diff(x_all)
        if np.any(dx <= 0):
            eps = 1e-9
            for i in range(1, len(x_all)):
                if x_all[i] <= x_all[i-1]:
                    x_all[i] = x_all[i-1] + eps

    # Build OpenFOAM point strings
    l11 = [format_point(x_all[i], y_top[i], 0.0)     for i in range(len(x_all))]  # top z=0
    l12 = [format_point(x_all[i], y_top[i], THICKNESS) for i in range(len(x_all))]  # top z=THICKNESS
    l21 = [format_point(x_all[i], y_bot[i], 0.0)     for i in range(len(x_all))]  # bottom z=0
    l22 = [format_point(x_all[i], y_bot[i], THICKNESS) for i in range(len(x_all))]  # bottom z=THICKNESS

    # Remove any consecutive duplicates (paranoia)
    l11 = dedupe_str_points(l11)
    l12 = dedupe_str_points(l12)
    l21 = dedupe_str_points(l21)
    l22 = dedupe_str_points(l22)

    return l11, l12, l21, l22, x_all, y_top, y_bot

# ------------------------------------------------
# Mesh writer
# ------------------------------------------------
def write_vertices_and_edges(path, x_all, y_top, y_bot, l11, l12, l21, l22):
    """
    Load the template blockMeshDict from 'path/system/blockMeshDict',
    replace the 'vertices' and 'edges' blocks entirely with consistent data,
    keep the rest of the file unchanged.
    """
    dict_path = os.path.join(path, "system", "blockMeshDict")
    with open(dict_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # --- Rebuild VERTICES block ---
    v0 = format_point(x_all[0],  y_bot[0 if len(y_bot)==0 else 0].split()[1][1:] if False else y_bot[0 if True else 0], 0.0)  # not used; we build directly below

    # explicit corners (match polyLine endpoints!)
    xmin = float(x_all[0])
    xmax = float(x_all[-1])
    ybot_L = float(y_bot[0])
    ybot_R = float(y_bot[-1])
    ytop_L = float(y_top[0])
    ytop_R = float(y_top[-1])

    vertices_block = []
    vertices_block.append("vertices")
    vertices_block.append("(")
    # z = 0
    vertices_block.append(f"    {format_point(xmin, ybot_L, 0.0)}")       # 0
    vertices_block.append(f"    {format_point(xmax, ybot_R, 0.0)}")       # 1
    vertices_block.append(f"    {format_point(xmax, ytop_R, 0.0)}")       # 2
    vertices_block.append(f"    {format_point(xmin, ytop_L, 0.0)}")       # 3
    # z = THICKNESS
    vertices_block.append(f"    {format_point(xmin, ybot_L, THICKNESS)}") # 4
    vertices_block.append(f"    {format_point(xmax, ybot_R, THICKNESS)}") # 5
    vertices_block.append(f"    {format_point(xmax, ytop_R, THICKNESS)}") # 6
    vertices_block.append(f"    {format_point(xmin, ytop_L, THICKNESS)}") # 7
    vertices_block.append(");")

    v_start, v_open, v_end = find_block(lines, "vertices")
    # Replace vertices block
    lines = lines[:v_start] + vertices_block + lines[v_end+1:]

    # --- Rebuild EDGES block ---
    edges_block = []
    edges_block.append("edges")
    edges_block.append("(")
    # bottom z=0: edge 0 1
    edges_block.append("\tpolyLine 0 1")
    edges_block.append("\t(")
    for pt in l21:
        edges_block.append("\t\t" + pt)
    edges_block.append("\t)")
    # bottom z=THICKNESS: edge 4 5
    edges_block.append("\tpolyLine 4 5")
    edges_block.append("\t(")
    for pt in l22:
        edges_block.append("\t\t" + pt)
    edges_block.append("\t)")
    # top z=0: edge 3 2
    edges_block.append("\tpolyLine 3 2")
    edges_block.append("\t(")
    for pt in l11:
        edges_block.append("\t\t" + pt)
    edges_block.append("\t)")
    # top z=THICKNESS: edge 7 6
    edges_block.append("\tpolyLine 7 6")
    edges_block.append("\t(")
    for pt in l12:
        edges_block.append("\t\t" + pt)
    edges_block.append("\t)")
    edges_block.append(");")

    e_start, e_open, e_end = find_block(lines, "edges")
    # Replace edges block
    lines = lines[:e_start] + edges_block + lines[e_end+1:]

    # Write back
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# ------------------------------------------------
# Main mesh pipeline
# ------------------------------------------------
def build_mesh(p1, p2, p3, path):
    # Copy template case
    shutil.copytree("swakless-1", path)

    # Build curves
    l11, l12, l21, l22, x_all, y_top, y_bot = build_arrays(p1, p2, p3)

    # Optional plot for quick visual confirmation
    if PLOT_GEOMETRY:
        plt.figure()
        plt.plot(x_all, y_top)
        plt.plot(x_all, y_bot)
        plt.xlim(min(x_all)-0.1, max(x_all)+0.1)
        plt.ylim(min(y_bot)-0.2, max(y_top)+0.2)
        plt.grid()
        plt.title(f"Reactor Geometry - p1={p1:.3f}, p2={p2:.3f}, p3={p3:.3f}")
        plt.savefig(os.path.join(path, "reactor_geometry.png"))
        plt.close()

    # Rebuild vertices + edges in blockMeshDict
    write_vertices_and_edges(path, x_all, y_top, y_bot, l11, l12, l21, l22)

    # Run blockMesh
    docker_run(['blockMesh'], path)

    # Run checkMesh and interpret results
    check_mesh_output = docker_run(['checkMesh'], path)
    out = check_mesh_output.stdout

    # Extract mesh size info
    mesh_size = None
    for line in out.splitlines():
        if line.strip().startswith("cells:"):
            try:
                mesh_size = int(line.split()[-1])
            except Exception:
                pass
            break

    # Decide if severe: delete or keep
    severe_patterns = [
        "Zero or negative face area",     # geometry collapse
        "Zero or negative cell volume",   # geometry collapse
        "incorrectly oriented",           # faces orientation fatal
    ]
    severe_hits = [p for p in severe_patterns if p in out]

    if severe_hits and DELETE_ON_SEVERE:
        print(f"[SEVERE] Deleting {path} due to:")
        for p in severe_hits:
            print(f"   - {p}")
        # Also print the offending lines for quick diagnosis
        for line in out.splitlines():
            if any(key in line for key in severe_patterns) or "Failed" in line:
                print("   " + line)
        shutil.rmtree(path)
        return None
    else:
        # Log non-severe quality issues (if any), but keep the mesh
        if "Failed" in out:
            print(f"[WARN] Mesh quality issues for {path}:")
            for line in out.splitlines():
                if line.strip().startswith("***") or "Failed" in line:
                    print("   " + line.strip())

    return mesh_size


def save_params_csv(rows, out_csv_path):
    """
    Save all generated samples to CSV:
    columns = p1, p2, p3, mesh_size, case_id, case_path
    mesh_size is empty if build failed / severe issues.
    """
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["p1", "p2", "p3", "mesh_size", "case_id", "case_path"])
        w.writerows(rows)

def docker_run(cmd, case_path, image="opencfd/openfoam-default:2506"):
    return subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{case_path}:/root",  
            "-w", "/root",
            image
        ] + cmd,
        capture_output=True, text=True
    )
    return result

# ------------------------------------------------
if __name__ == "__main__":
    mesh_sizes = []

    ap = argparse.ArgumentParser(description="Collect ONLY valid (p1,p2,p3) for VAE.")
    ap.add_argument("--out-root", required=True, help="Directory where the CSV dataset will be saved")
    ap.add_argument("--base-case-dir", default="./swakless-1")
    ap.add_argument("--docker-image", default="opencfd/openfoam-default:2506")  # kept for compatibility (unused)
    ap.add_argument("--n-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p1-min", type=float, default=0.1)
    ap.add_argument("--p1-max", type=float, default=0.7)
    ap.add_argument("--p2-min", type=float, default=3.0)
    ap.add_argument("--p2-max", type=float, default=6.0)
    args = ap.parse_args()

    # Ensure output root exists
    out_root = os.path.abspath(os.path.expanduser(args.out_root))
    os.makedirs(out_root, exist_ok=True)

    # Sampling
    num_samples = args.n_samples
    num_dimensions = 3
    sampler = qmc.LatinHypercube(d=num_dimensions, seed=args.seed)
    sample = sampler.random(n=num_samples)

    lower_bounds = [max(0.1, args.p1_min), max(3.0, args.p2_min), 0.0]
    upper_bounds = [min(0.5,  args.p1_max), min(6.0,  args.p2_max), math.pi/2]

    scaled_sample = qmc.scale(sample, lower_bounds, upper_bounds)

    # Collect rows for CSV
    csv_rows = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv_path = os.path.join(out_root, f"vae_params_{timestamp}.csv")

    for i in range(num_samples):
        p1, p2, p3 = map(float, scaled_sample[i])
        case_id = "{}_{}".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), uuid.uuid4().hex)
        # Keep your original per-case location (unchanged)
        case_path = os.path.join(
            os.path.expanduser("~/foam/4th-Year-Research-Project/2D_Reactor/generate_mesh_1"),
            case_id
        )

        try:
            mesh_size = build_mesh(p1, p2, p3, case_path)
        except Exception as e:
            print(f"[ERROR] Build failed for {case_path}: {e}")
            mesh_size = None
            # cleanup partial folder to avoid clutter
            if os.path.isdir(case_path):
                shutil.rmtree(case_path)

        mesh_sizes.append(mesh_size)
        # Append to CSV rows (mesh_size is None -> empty cell)
        csv_rows.append([f"{p1:.8f}", f"{p2:.8f}", f"{p3:.8f}",
                         "" if mesh_size is None else int(mesh_size),
                         case_id, case_path])

    # Save the dataset CSV with ALL generated samples
    save_params_csv(csv_rows, out_csv_path)

    print(mesh_sizes)
    print(f"\nSaved ALL sampled parameters (including failed meshes) to:\n  {out_csv_path}")
