#!/usr/bin/env python3
"""
Plot raw inlet/outlet time series (first & last internalField entries) for p, |U|, and s
for every subfolder inside ROOT_DIR, starting at t >= 0.05 up to 5.0.

Reads p, phi, s, U, vorticityField to honor file presence, but ONLY plots p, |U|, s.

No CSVs. PNGs saved to <case>/plots_inlet_outlet/.
"""

import re
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ensure no GUI backend
import matplotlib.pyplot as plt

# -------------------- USER CONFIG --------------------
ROOT_DIR = Path("/home/ts322/ResearchProject/4th-Year-Research-Project/2D_Reactor/Results_2025-10-01_14-46")
T_MIN = 0.05
T_MAX = 5.0
FIELDS = ["p", "phi", "s", "U", "vorticityField"]  # Plotting only p, s, U
SAVE_PLOTS = True
SHOW_PLOTS = False  # <-- save only, no interactive windows
# -----------------------------------------------------


def is_floatlike_dirname(name: str) -> bool:
    return re.fullmatch(r"[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?", name) is not None


def parse_internal_field_text(txt: str):
    """
    Parse OpenFOAM internalField from a field file.
    Supports:
      - internalField uniform <scalar>;
      - internalField uniform (<Ux> <Uy> <Uz>);
      - internalField nonuniform List<scalar> N ( ... );
      - internalField nonuniform List<vector> N ( (x y z) ... );
    Returns:
      - numpy array of shape (N,) for scalars, or (N, 3) for vectors
      - a parse mode string (for basic debugging)
    """
    # Uniform (scalar or vector)
    m_uniform = re.search(r"internalField\s+uniform\s+([^;]+);", txt)
    if m_uniform:
        val_str = m_uniform.group(1).strip()
        if val_str.startswith("(") and val_str.endswith(")"):
            parts = re.split(r"[ \t]+", val_str.strip("() ").strip())
            if len(parts) == 3:
                try:
                    vec = np.array([[float(parts[0]), float(parts[1]), float(parts[2])]])
                    return vec, "vector_uniform"
                except Exception:
                    return None, "vector_uniform_parse_error"
            return None, "vector_uniform_malformed"
        else:
            try:
                val = float(val_str)
                return np.array([val]), "scalar_uniform"
            except Exception:
                return None, "scalar_uniform_parse_error"

    # Nonuniform (scalar or vector list)
    m_nonuni = re.search(
        r"internalField\s+nonuniform\s+List<([^>]+)>\s+([0-9]+)\s*\((.*?)\)\s*;",
        txt, flags=re.S
    )
    if m_nonuni:
        typ = m_nonuni.group(1).strip().lower()
        N = int(m_nonuni.group(2))
        body = m_nonuni.group(3).strip()
        if typ == "scalar":
            vals = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", body)
            try:
                scalars = np.array([float(v) for v in vals[:N]])
                return scalars, "scalar_list"
            except Exception:
                return None, "scalar_list_parse_error"
        elif typ == "vector":
            vecs = re.findall(
                r"\(\s*([-+]?[\d\.Ee+-]+)\s+([-+]?[\d\.Ee+-]+)\s+([-+]?[\d\.Ee+-]+)\s*\)",
                body
            )
            try:
                vec_arr = np.array([[float(a), float(b), float(c)] for a, b, c in vecs[:N]])
                return vec_arr, "vector_list"
            except Exception:
                return None, "vector_list_parse_error"

    return None, "unparsed"


def read_field(field_path: Path):
    """Read and parse a field file; returns (array, parse_mode)."""
    if not field_path.exists():
        return None, "missing"
    try:
        txt = field_path.read_text(errors="ignore")
    except Exception as e:
        return None, f"read_error:{e!r}"
    return parse_internal_field_text(txt)


def first_last_scalar(arr: np.ndarray):
    """(first, last) for a 1D scalar array or flattened array."""
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return float(arr[0]), float(arr[-1])


def first_last_vector_mag(vecs: np.ndarray):
    """(first_mag, last_mag) for an (N,3) vector array (or a single 3-vector)."""
    if vecs.ndim == 1 and vecs.size == 3:
        mag = float(np.linalg.norm(vecs))
        return mag, mag
    mags = np.linalg.norm(vecs, axis=1)
    return float(mags[0]), float(mags[-1])


def collect_case_raw(case_dir: Path):
    """
    For a single case directory (containing time subfolders), read fields at each time
    and collect raw inlet/outlet values:
      - Scalars: p, phi, s, vorticityField => first & last entries
      - Vector: U => first & last magnitudes
    Returns dict: { field_name: {"time": [...], "inlet": [...], "outlet": [...]}, ... }
    """
    # Identify time directories
    time_dirs = []
    for entry in case_dir.iterdir():
        if entry.is_dir() and is_floatlike_dirname(entry.name):
            t_val = float(entry.name)
            if T_MIN <= t_val <= T_MAX:
                time_dirs.append((t_val, entry))
    time_dirs.sort(key=lambda x: x[0])

    result = {f: {"time": [], "inlet": [], "outlet": []} for f in FIELDS}

    for t, tdir in time_dirs:
        for fname in FIELDS:
            fpath = tdir / fname
            data, _ = read_field(fpath)
            if data is None:
                continue
            if fname == "U":
                inlet, outlet = first_last_vector_mag(data)
            else:
                inlet, outlet = first_last_scalar(data)
            result[fname]["time"].append(t)
            result[fname]["inlet"].append(inlet)
            result[fname]["outlet"].append(outlet)

    return result


def plot_three(case_dir: Path, series: dict):
    """
    Create 3 plots for a case: p, |U|, s (inlet vs outlet vs time).
    Saves into case_dir / "plots_inlet_outlet".
    """
    out_dir = case_dir / "plots_inlet_outlet"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _plot_one(field_key: str, ylabel: str, filename: str):
        if field_key not in series:
            return
        times = series[field_key]["time"]
        inlet = series[field_key]["inlet"]
        outlet = series[field_key]["outlet"]
        if not times:
            return
        fig = plt.figure()
        plt.plot(times, inlet, label="inlet (first value)")
        plt.plot(times, outlet, label="outlet (last value)")
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} at inlet/outlet (raw) — {case_dir.name}")
        plt.legend()
        fig.savefig(out_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    _plot_one("p", "p (raw)", "p_inlet_outlet_vs_time.png")
    _plot_one("U", "|U| (raw magnitude)", "U_inlet_outlet_vs_time.png")
    _plot_one("s", "s (raw)", "s_inlet_outlet_vs_time.png")


def find_case_dirs(root: Path) -> List[Path]:
    """
    Return a list of immediate subdirectories that look like 'case' dirs
    (contain at least one numeric time folder). If none are found,
    treat the root itself as a single case directory.
    """
    candidates = [p for p in root.iterdir() if p.is_dir()]
    case_dirs = []
    for c in candidates:
        has_time = any(is_floatlike_dirname(ch.name) for ch in c.iterdir() if ch.is_dir())
        if has_time:
            case_dirs.append(c)
    if not case_dirs:
        case_dirs = [root]
    return case_dirs


def main():
    root = ROOT_DIR
    if not root.exists():
        raise SystemExit(f"[ERROR] ROOT_DIR does not exist: {root}")

    case_dirs = find_case_dirs(root)
    print(f"[INFO] Found {len(case_dirs)} case dir(s) under: {root}")

    for case in case_dirs:
        print(f"[INFO] Processing case: {case}")
        series = collect_case_raw(case)
        plot_three(case, series)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
