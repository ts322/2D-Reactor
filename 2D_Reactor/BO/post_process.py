import numpy as np
import math
import re
from scipy.optimize import curve_fit

def compute_N(scalar_transport_path: str, data_path: str):
    """
    Compute Tanks-in-Series parameters from an OpenFOAM case:
    - N_moment  (τ² / σ_t²)
    - N_curve   (nonlinear fit of E(θ))
    Also returns τ (mean residence time).

    Parameters
    ----------
    scalar_transport_path : str
        Path to 'system/scalarTransport' (used only for sanity/future use).
    data_path : str
        Path to tracer outlet data file with two columns: t, s(t).

    Returns
    -------
    N_moment : float
    N_curve  : float (np.nan if fit fails)
    tau      : float
    """

    # -------- read scalarTransport (optional, kept for future extensions)
    def read_scalar_transport(path):
        params = {}
        try:
            with open(path, "r") as f:
                for line in f:
                    m = re.match(r"^\s*(\w+)\s+([\d\.Ee+-]+);", line)
                    if m:
                        k, v = m.groups()
                        params[k] = float(v)
        except FileNotFoundError:
            pass
        return params

    _ = read_scalar_transport(scalar_transport_path)

    # -------- load tracer data
    # expects two columns: time, signal
    t, s = np.loadtxt(data_path, comments="#", unpack=True)
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)

    # guard against negatives / NaNs in signal
    mask = np.isfinite(t) & np.isfinite(s) & (s >= 0.0)
    t, s = t[mask], s[mask]
    # ensure strictly increasing time
    order = np.argsort(t)
    t, s = t[order], s[order]

    # -------- exit-age distribution E(t) = s / ∫ s dt
    s_int = np.trapz(s, t)
    if s_int <= 0:
        raise ValueError("Integral of signal is non-positive; cannot normalize.")
    E_t = s / s_int

    # -------- mean residence time τ = ∫ t E(t) dt
    tau = np.trapz(t * E_t, t)

    # -------- variance σ_t² = ∫ (t-τ)² E(t) dt
    sigma_t_sq = np.trapz(((t - tau) ** 2) * E_t, t)
    if sigma_t_sq <= 0:
        # degeneracy; return moment N as inf
        N_moment = np.inf
    else:
        N_moment = (tau ** 2) / sigma_t_sq

    # -------- TiS model: E(θ; N)
    def E_TiS(theta, N):
        # E(θ) = [N * (Nθ)^(N-1) / Γ(N)] * exp(-Nθ)
        # clip to avoid inf at θ=0 when N<1 (we bound N≥1 anyway)
        theta = np.clip(theta, 0, None)
        return (N * (N * theta) ** (N - 1) * np.exp(-N * theta)) / math.gamma(N)

    # -------- build θ-data
    if tau <= 0:
        N_curve = np.nan
    else:
        theta = t / tau
        E_theta = tau * E_t  # E(θ) = τ E(t)

        # fit only where E_theta is positive & finite
        m_fit = np.isfinite(theta) & np.isfinite(E_theta) & (E_theta >= 0)
        theta_fit = theta[m_fit]
        E_fit = E_theta[m_fit]

        # tiny regularization to avoid zero-only data
        if theta_fit.size < 5 or np.all(E_fit == 0):
            N_curve = np.nan
        else:
            # heuristic initial guess from moment estimate (clamped)
            p0 = max(1.0, min(float(N_moment) if np.isfinite(N_moment) else 3.0, 200.0))
            try:
                popt, _ = curve_fit(
                    E_TiS,
                    theta_fit,
                    E_fit,
                    p0=[p0],
                    bounds=(1.0, 200.0),
                    maxfev=20000,
                )
                N_curve = float(popt[0])
            except Exception:
                N_curve = np.nan

    return float(N_moment), float(N_curve) if np.isfinite(N_curve) else np.nan, float(tau)
