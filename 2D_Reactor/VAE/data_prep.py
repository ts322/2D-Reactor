#!/usr/bin/env python3
"""
Data preparation script for VAE training on reactor geometries using Dockerized OpenFOAM.

This script:
  - Samples (p1, p2, p3) using Latin Hypercube Sampling
  - Builds OpenFOAM meshes via a Docker container
  - Checks mesh validity
  - Saves valid parameter sets in a CSV

Example usage:
  python data_prep.py \
    --samples 50 \
    --mesh-root ~/ResearchProject/4th-Year-Research-Project/2D_Reactor/generate_mesh \
    --out ~/ResearchProject/4th-Year-Research-Project/2D_Reactor/datasets/run_$(date +%Y%m%d_%H%M%S)/vae_params.csv \
    --docker-image opencfd/openfoam-default:2506 \
    --plot-geometry
"""

import os
import sys
import csv
import math
import uuid
import shutil
import argparse
import subprocess
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np
from scipy import interpolate
from scipy.stats import qmc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ---------------------------- Constants ----------------------------
THICKNESS_DEFAULT = 0.1
TARGET_END_WIDTH_DEFAULT = 0.5
MIN_GAP_DEFAULT = 0.02
N_ADD_DEFAULT = 50
BASE_N_DEFAULT = 180
SEVERE_PATTERNS = ["Zero or negative face area", "Zero or negative cell volume", "incorrectly oriented"]

# ---------------------------- Helpers ----------------------------
def sin_line(x, b, a, c, off): return b + (a * np.sin(c * (x - off)))

def smooth_local(x, y, idx, k=5):
    i0 = max(idx - k, 0); i2 = min(idx + k, len(x)-1)
    xm, ym = [x[i0], x[idx], x[i2]], [y[i0], ((y[i0] + y[i2]) * 0.5 + y[idx]) * 0.5, y[i2]]
    x_new = np.linspace(x[i0], x[i2], (i2 - i0) + 1)
    y_new = interp1d(xm, ym, kind='quadratic')(x_new)
    x[i0:i2+1], y[i0:i2+1] = x_new, y_new
    return x, y

def format_point(x, y, z): return f"({x:.8f} {y:.8f} {z:.8f})"

def dedupe(seq):
    out, last = [], None
    for s in seq:
        if s != last: out.append(s); last = s
    return out

def find_block(lines, key):
    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith(key)), None)
    if start is None: raise RuntimeError(f"Couldn't find '{key}' block in blockMeshDict")
    open_idx = start if "(" in lines[start] else start + 1
    end_idx = next((j for j in range(open_idx, len(lines)) if lines[j].strip() == ");"), None)
    if end_idx is None: raise RuntimeError(f"Couldn't find end of '{key}' block (');')")
    return start, open_idx, end_idx

# ---------------------------- Geometry builder ----------------------------
def build_arrays(p1, p2, p3, thickness, target_end_width, min_gap, n_add, base_n):
    if not (0.1 <= p1 <= 0.5): raise ValueError(f"p1 out of range: {p1}")
    if not (3.0 <= p2 <= 6.0): raise ValueError(f"p2 out of range: {p2}")
    if not (0.0 <= p3 <= np.pi/2): raise ValueError(f"p3 out of range: {p3}")

    x_core = np.linspace(0.0, 3.0, base_n)
    y_top_core = sin_line(x_core, 0.5, p1, p2, p3)
    y_bot_core = sin_line(x_core, 0.0, 0.25, p2, p3)
    add_start_x = np.linspace(x_core[0] - 1.0, x_core[0], n_add + 1)[:-1]
    add_end_x = np.linspace(x_core[-1], x_core[-1] + 1.0, n_add + 1)[1:]

    y_top_start_L = y_bot_core[0] + target_end_width
    y_top_end_R = y_bot_core[-1] + target_end_width
    f_y1_start = interpolate.interp1d([add_start_x[0], add_start_x[-1]], [y_top_start_L, y_top_core[0]])
    f_y1_end = interpolate.interp1d([add_end_x[0], add_end_x[-1]], [y_top_core[-1], y_top_end_R])

    y_top = np.concatenate([f_y1_start(add_start_x), y_top_core, f_y1_end(add_end_x)])
    y_bot = np.concatenate([np.full_like(add_start_x, y_bot_core[0]), y_bot_core, np.full_like(add_end_x, y_bot_core[-1])])
    x_all = np.concatenate([add_start_x, x_core, add_end_x])

    for idx in [n_add, n_add + base_n]:
        x_all, y_top = smooth_local(x_all, y_top, idx)
        x_all, y_bot = smooth_local(x_all, y_bot, idx)

    y_top = np.maximum(y_top, y_bot + min_gap)
    order = np.argsort(x_all)
    x_all, y_top, y_bot = x_all[order], y_top[order], y_bot[order]

    l11 = [format_point(x_all[i], y_top[i], 0.0) for i in range(len(x_all))]
    l12 = [format_point(x_all[i], y_top[i], thickness) for i in range(len(x_all))]
    l21 = [format_point(x_all[i], y_bot[i], 0.0) for i in range(len(x_all))]
    l22 = [format_point(x_all[i], y_bot[i], thickness) for i in range(len(x_all))]
    return dedupe(l11), dedupe(l12), dedupe(l21), dedupe(l22), x_all, y_top, y_bot

# ---------------------------- Docker OpenFOAM utilities ----------------------------
def docker_run(cmd: List[str], case_path: str, image: str = "opencfd/openfoam-default:2506") -> subprocess.CompletedProcess:
    """Run an OpenFOAM command inside a Docker container."""
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.path.abspath(case_path)}:/home/openfoam/case",
        "-w", "/home/openfoam/case", image
    ] + cmd
    return subprocess.run(docker_cmd, capture_output=True, text=True)

def parse_checkmesh_output(text):
    mesh_size = None
    for line in text.splitlines():
        if line.strip().startswith("cells:"):
            try: mesh_size = int(line.split()[-1])
            except: pass
    severe_hits = [p for p in SEVERE_PATTERNS if p in text]
    failed = ("Failed" in text) or bool(severe_hits)
    return not failed, mesh_size, severe_hits

# ---------------------------- Mesh generation ----------------------------
def build_mesh(p1, p2, p3, case_root, image, thickness, target_end_width, min_gap, n_add, base_n, plot_geom, delete_on_severe):
    case_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}"
    case_path = os.path.join(os.path.expanduser(case_root), case_id)
    shutil.copytree("Example", case_path)

    l11, l12, l21, l22, x_all, y_top, y_bot = build_arrays(p1, p2, p3, thickness, target_end_width, min_gap, n_add, base_n)
    if plot_geom:
        plt.figure()
        plt.plot(x_all, y_top); plt.plot(x_all, y_bot)
        plt.title(f"p1={p1:.2f}, p2={p2:.2f}, p3={p3:.2f}")
        plt.savefig(os.path.join(case_path, "geometry.png")); plt.close()

    # Run blockMesh + checkMesh in Docker
    bmesh = docker_run(["blockMesh"], case_path, image=image)
    if bmesh.returncode != 0:
        print(f"[ERROR] blockMesh failed for {case_id}")
        shutil.rmtree(case_path, ignore_errors=True)
        return False, None, case_path, ["blockMesh failed"]

    cmesh = docker_run(["checkMesh"], case_path, image=image)
    is_valid, mesh_size, severe_hits = parse_checkmesh_output(cmesh.stdout)
    if (not is_valid) and delete_on_severe:
        print(f"[SEVERE] Deleting {case_path} due to {severe_hits}")
        shutil.rmtree(case_path, ignore_errors=True)
    return is_valid, mesh_size, case_path, severe_hits

# ---------------------------- Main ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate VAE training data with Docker-based OpenFOAM.")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--docker-image", default="opencfd/openfoam-default:2506")
    parser.add_argument("--plot-geometry", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delete-on-severe", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(os.path.expanduser(args.mesh_root), exist_ok=True)
    param_array = qmc.scale(qmc.LatinHypercube(d=3, seed=args.seed).random(n=args.samples),
                            [0.1, 3.0, 0.0], [0.5, 6.0, math.pi/2])

    results = []
    for i, (p1, p2, p3) in enumerate(param_array, 1):
        print(f"[{i}/{args.samples}] Building mesh for ({p1:.3f}, {p2:.3f}, {p3:.3f})...")
        ok, cells, case_path, issues = build_mesh(
            p1, p2, p3, args.mesh_root, args.docker_image,
            THICKNESS_DEFAULT, TARGET_END_WIDTH_DEFAULT, MIN_GAP_DEFAULT,
            N_ADD_DEFAULT, BASE_N_DEFAULT, args.plot_geometry, args.delete_on_severe
        )
        results.append({"p1": p1, "p2": p2, "p3": p3, "valid": int(ok), "cells": cells or "", "case_path": case_path})
        print(f"  → {'OK' if ok else 'FAIL'} ({cells})")

    os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)
    with open(os.path.expanduser(args.out), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["p1", "p2", "p3", "valid", "cells", "case_path"])
        writer.writeheader(); writer.writerows(results)

    print(f"Saved dataset to {args.out}")

if __name__ == "__main__":
    main()
