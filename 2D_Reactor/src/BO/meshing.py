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
    l11, l12, l21, l22, x_all, y_top, y_bot = build_arrays(p1, p2, p3)
    image = "opencfd/openfoam-default:2506"
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

    write_vertices_and_edges(path, x_all, y_top, y_bot, l11, l12, l21, l22)

    bmesh =docker_run(["blockMesh"], path, image=image)

    cmesh = docker_run(["checkMesh"], path, image=image)
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