
#!/usr/bin/env python3
"""
Minimal BoTorch scaffold over a latent space.

- Defines latent-space bounds (default [-3, 3] per dimension).
- Draws Sobol initialization points.
- Fits a GP and runs qNEI to propose candidates.
- Calls a user-provided objective function: objective(z) -> float.

You implement objective(z) yourself (e.g., decode -> build mesh -> run solver -> return metric).
This keeps your existing objective code untouched — just import or paste it below.
"""
import os, csv, argparse, random
import numpy as np
import torch
import subprocess, shlex
import math 

import torch
import torch.nn as nn
import torch.nn.functional as F
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


from meshing import build_mesh
from post_process import compute_N
# Load pretrained VAE

# --- near top of file, after imports ---

from ae import AE
from meshing import build_mesh
from post_process import compute_N

# Globals that objective uses (we'll set these in main())
vae_model = None
base_dir = os.path.expanduser("~/2D-Reactor/2D_Reactor")  # adjust if your project root differs

# -------------------------
# User objective function
# -------------------------
def objective(z: torch.Tensor, iters: int = None) -> float:
    """
    z: torch.Tensor (latent vector)
    iters: optional integer iteration id used to label output files/dirs
    returns: scalar metric (float)
    """
    global vae_model, base_dir
    if vae_model is None:
        raise RuntimeError("vae_model not loaded. Make sure to load the VAE before calling objective().")

       # --- decode z robustly and map to valid parameter ranges ---
    # ensure batch shape
    z_in = z.unsqueeze(0) if z.ndim == 1 else z
    z_in = z_in.float()

    with torch.no_grad():
        decoded = vae_model.decoder(z_in)

    # collapse batch if single
    if decoded.ndim == 2 and decoded.shape[0] == 1:
        decoded = decoded.squeeze(0)

    # Bring to numpy
    p_vals = decoded.detach().cpu().numpy()

    # Try to extract 3 scalar parameters.
    # If decoder returns a vector longer than 3, assume first 3 are p1,p2,p3.
    # If it returns fewer than 3, fail early.
    if p_vals.ndim == 0:
        # single scalar, impossible for 3 params
        raise RuntimeError(f"Decoder returned a scalar, expected >=3 values.")
    elif p_vals.ndim == 1:
        if p_vals.size < 3:
            raise RuntimeError(f"Decoder returned {p_vals.size} values, expected >=3.")
        raw_p = p_vals[:3].astype(float)
    else:
        # e.g., (N, D) but we already collapsed batch; defensive:
        raw_p = np.asarray(p_vals).reshape(-1)[:3].astype(float)

    # Define target ranges
    p1_lo, p1_hi = 0.1, 0.5
    p2_lo, p2_hi = 3.0, 6.0
    p3_lo, p3_hi = 0.0, math.pi / 2.0

    # Helper: map arbitrary real -> [lo,hi]
    def map_to_range(x, lo, hi):
        x = float(x)
        if not np.isfinite(x):
            return None
        # If x looks already in [0,1], just scale
        if 0.0 <= x <= 1.0:
            s = x
        # If x looks in [-1,1], shift to [0,1]
        elif -1.0 <= x <= 1.0:
            s = (x + 1.0) * 0.5
        else:
            # otherwise squash with sigmoid to [0,1]
            s = 1.0 / (1.0 + math.exp(-x))
        return lo + s * (hi - lo)

    p1_m = map_to_range(raw_p[0], p1_lo, p1_hi)
    p2_m = map_to_range(raw_p[1], p2_lo, p2_hi)
    p3_m = map_to_range(raw_p[2], p3_lo, p3_hi)

    # If any mapping failed (NaN/inf), bail with fallback or raise
    if p1_m is None or p2_m is None or p3_m is None:
        print(f"[WARN] Decoder produced invalid values: raw={raw_p}")
        # choose to return a bad metric so BO avoids this point
        return float("-1e9")

    # Final clipping just in case
    p1 = float(np.clip(p1_m, p1_lo, p1_hi))
    p2 = float(np.clip(p2_m, p2_lo, p2_hi))
    p3 = float(np.clip(p3_m, p3_lo, p3_hi))

    # Debug logging (you can remove or lower verbosity later)
    print(f"[DEBUG] decoded raw: {raw_p}  -> mapped p1={p1:.6f}, p2={p2:.6f}, p3={p3:.6f}")

    # iteration id for naming outputs
    iter_val = (iters + 1) 
    ID = f"iter_{iter_val}"
    path = os.path.join(base_dir, "Mesh", ID)
   

    # build mesh (your function)
    build_mesh(p1, p2, p3, path)

    # call external script (run_it.sh) - ensure it prints the run directory on stdout
    script = os.path.join(base_dir, "BO", "run_it.sh")
    proc = subprocess.run([f"{script}", f"{path}", f'{iter_val}'],
                      capture_output=True, text=True, check=True)
    run_dir = proc.stdout.strip().splitlines()[-1]
    print("Run dir:", run_dir)
    data_path = os.path.join(run_dir,'postProcessing/surfaceFieldValue1/0/surfaceFieldValue_0.dat')
    scalar_path = os.path.join(run_dir, 'system/scalarTransport')
    
    N_moment, N_curve, tau = compute_N(scalar_path, data_path)
    print(f"{N_curve}")
    return float(N_curve)

def robust_load_state_dict(vae_model, ckpt_path):
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    candidates = []

    # Candidate 1: checkpoint itself (sometimes it's already a raw state_dict)
    if isinstance(ckpt, dict):
        candidates.append(ckpt)
        # Common nested keys that may contain the actual state_dict
        for k in ("model_state_dict", "state_dict", "model_state", "model", "state"):
            if k in ckpt and isinstance(ckpt[k], dict):
                candidates.append(ckpt[k])
    else:
        # saved object (rare)
        candidates.append(ckpt)

    # Try direct loads first
    for idx, cand in enumerate(candidates):
        try:
            vae_model.load_state_dict(cand)
            print(f"[loader] Loaded checkpoint using candidate #{idx} (direct).")
            return
        except Exception as e:
            # store exception for later debug
            pass

    # Try stripping common prefixes
    prefixes = ["module.", "vae_model.", "model.", "net."]
    for idx, cand in enumerate(candidates):
        for pref in prefixes:
            stripped = {k.replace(pref, ""): v for k, v in cand.items()}
            try:
                vae_model.load_state_dict(stripped)
                print(f"[loader] Loaded checkpoint using candidate #{idx} after stripping prefix '{pref}'.")
                return
            except Exception:
                continue

    # If we get here, loading failed. Print diagnostics to help map keys.
    expected = set(vae_model.state_dict().keys())
    # pick the first dict-like candidate to inspect
    provided = set()
    for cand in candidates:
        if isinstance(cand, dict):
            provided = set(cand.keys())
            break

    print("\n[loader] FAILED to load state_dict automatically.")
    print(f"[loader] expected (sample 20): {list(expected)[:20]}")
    print(f"[loader] provided (sample 40): {list(provided)[:40]}")
    print(f"[loader] #expected keys = {len(expected)}, #provided keys = {len(provided)}")
    missing = sorted(list(expected - provided))
    unexpected = sorted(list(provided - expected))
    print(f"[loader] #missing keys = {len(missing)}, #unexpected keys = {len(unexpected)}")
    print(f"[loader] missing (sample 20): {missing[:20]}")
    print(f"[loader] unexpected (sample 20): {unexpected[:20]}")

    # Helpful hint to user
    raise RuntimeError("Could not auto-load checkpoint. Inspect the printed key lists above; "
                       "if the checkpoint contains its model under a different nested key, "
                       "or uses different name prefixes, adapt the loader accordingly.")


# BO loop
def bo_loop(latent_dim: int, iters: int, batch_size: int, init_sobol: int, results_out: str,
            seed: int = 123, maximize: bool = True, lo: float = -5, hi: float = 5):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Bounds
    bounds = torch.stack([torch.full((latent_dim,), lo), torch.full((latent_dim,), hi)]).double()

    # Initial Sobol points
    sobol = torch.quasirandom.SobolEngine(dimension=latent_dim, scramble=True, seed=seed)
    X0_unit = sobol.draw(init_sobol)             # [0,1]^d
    X0 = bounds[0] + (bounds[1] - bounds[0]) * X0_unit.double()  # scale to [lo,hi]

    # Evaluate initial design
    X = []
    y = []
    logs = []
    for i in range(init_sobol):
        z = X0[i].detach().double()
        val = float(objective(z, iters=len(logs)))
        X.append(z.unsqueeze(0))
        y.append(torch.tensor([[val]], dtype=torch.double))
        logs.append({"iter": i, "z": z.tolist(), "metric": val})
        print(f"[Init {i+1}/{init_sobol}] metric={val:.6f}")

    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    # BO iterations
    for t in range(iters):
        # Fit GP
        model = SingleTaskGP(X, y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Acquisition: qNEI
        best_f = y.max() if maximize else y.min()
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acqf = qNoisyExpectedImprovement(
            model=model,
            X_baseline=X,      # keep this as the same X you used for training the GP
            sampler=sampler,
            prune_baseline=True,
        )

        # Optimize acquisition
        cand, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=batch_size,
            num_restarts=5,
            raw_samples=128,
            options={"batch_limit": 5, "maxiter": 200},
        )

        # Evaluate batch
        for j in range(batch_size):
            z = cand[j].detach().double()
            val = float(objective(z, iters=len(logs)))
            X = torch.cat([X, z.view(1, -1)], dim=0)
            y = torch.cat([y, torch.tensor([[val]], dtype=torch.double)], dim=0)
            logs.append({"iter": len(logs), "z": z.tolist(), "metric": val})
            print(f"[Iter {t+1}/{iters} • {j+1}/{batch_size}] metric={val:.6f}")

    # Save results
    os.makedirs(os.path.dirname(os.path.expanduser(results_out)), exist_ok=True)
    with open(os.path.expanduser(results_out), "w", newline="") as f:
        fields = ["iter", "metric", "z"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in logs:
            w.writerow({k: row.get(k, "") for k in fields})

    # Report best
    metrics = np.array([r["metric"] for r in logs], dtype=float)
    best_idx = int(np.nanargmax(metrics)) if maximize else int(np.nanargmin(metrics))
    best = logs[best_idx]
    print(f"[BEST] metric={best['metric']:.6f} at z={best['z']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-dim", "--latent_dim", dest="latent_dim",
                    type=int, required=True,
                    help="Latent dimension (accepts --latent-dim or --latent_dim)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--init-sobol", type=int, default=8)
    ap.add_argument("--results-out", type=str, default="./bo_runs/base_run.csv")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--minimize", action="store_true")
    ap.add_argument("--lo", type=float, default=-3.0, help="Lower bound per latent dim")
    ap.add_argument("--hi", type=float, default=3.0, help="Upper bound per latent dim")
    args = ap.parse_args()

    # ---- Declare globals immediately (must come before any use of vae_model) ----
    global vae_model, base_dir
    base_dir = os.path.expanduser("~/2D-Reactor/2D_Reactor")
    vae_model = None

    # ---- Load VAE checkpoint and instantiate model ----
    vae_path = os.path.expanduser('/Users/marcobarbacci/2D-Reactor/2D_Reactor/VAE/Weights/vae_model.pt')
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"VAE checkpoint not found at: {vae_path}")

    ckpt = torch.load(vae_path, map_location=torch.device('cpu'))

    # Try to locate a state-dict inside the checkpoint under common keys
    state = None
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict", "model_state", "model", "state"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        # fallback: maybe ckpt itself is a state-dict (contains tensor values)
        if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
    else:
        # ckpt is possibly a saved model object
        vae_model = ckpt

    # instantiate model (use args.latent_dim so decoder shape matches)
    if vae_model is None:
        vae_model = AE(latent_dim=args.latent_dim)

    # attempt to load weights if we found a state-dict
    if state is not None:
        try:
            vae_model.load_state_dict(state)
        except RuntimeError:
            # try stripping common prefixes (e.g., 'module.' from DataParallel)
            stripped = {k.replace("module.", ""): v for k, v in state.items()}
            vae_model.load_state_dict(stripped)

    vae_model.eval()
    # ---------------------------------------------------------------------------

    # run the BO loop
    bo_loop(
        latent_dim=args.latent_dim,
        iters=args.iters,
        batch_size=args.batch_size,
        init_sobol=args.init_sobol,
        results_out=args.results_out,
        seed=args.seed,
        maximize=not args.minimize,
        lo=args.lo,
        hi=args.hi,
    )
#TODO potentially consider taking the final iteration and running it through paraview to viualise the results

if __name__ == "__main__":
    main()
