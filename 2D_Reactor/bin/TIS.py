import numpy as np
import matplotlib.pyplot as plt
import re
import math
from scipy.optimize import curve_fit

# --------------------------------------------------------
# 1️⃣ Read timing parameters from controlDict
# --------------------------------------------------------
control_dict_path = "/home/ts322/ResearchProject/4th-Year-Research-Project/2D_Reactor/Results/Results_2025-10-08_15-18/system/controlDict"

def read_control_dict(path):
    params = {}
    with open(path, 'r') as f:
        for line in f:
            match = re.match(r"^\s*(\w+)\s+([\d\.Ee+-]+);", line)
            if match:
                key, value = match.groups()
                params[key] = float(value)
    return params

params = read_control_dict(control_dict_path)
deltaT = params.get('deltaT', 0.01)
writeInterval = params.get('writeInterval', 20)
endTime = params.get('endTime', 5)

print(f"deltaT = {deltaT}")
print(f"writeInterval = {writeInterval}")
print(f"endTime = {endTime}")
print(f"Expected write frequency: {writeInterval * deltaT:.3f} s")

# --------------------------------------------------------
# 2️⃣ Read tracer outlet data
# --------------------------------------------------------
datapath = "/home/ts322/ResearchProject/4th-Year-Research-Project/2D_Reactor/Results/Results_2025-10-08_15-18/postProcessing/surfaceFieldValue1/0/surfaceFieldValue_0.dat"
t, s = np.loadtxt(datapath, comments="#", unpack=True)
s = np.abs(s)  # Remove any negative noise

# --------------------------------------------------------
# 3️⃣ Compute Δt and E(t)
# --------------------------------------------------------
dt = np.mean(np.diff(t))
area = np.sum(s * dt)
E_t = s / area
print(f"Normalisation check (∑E(t)Δt ≈ 1): {np.sum(E_t * dt):.6f}")

# --------------------------------------------------------
# 4️⃣ Compute τ, θ, and E(θ)
# --------------------------------------------------------
tau = np.sum(t * E_t * dt)
theta = t / tau
E_theta = tau * E_t
print(f"Mean residence time τ = {tau:.6f} s")

# --------------------------------------------------------
# 5️⃣ Define TiS model for curve fitting
# --------------------------------------------------------
def E_TiS(theta, N):
    N = np.maximum(N, 1e-6)  # avoid invalid values
    return (N**N * theta**(N - 1) * np.exp(-N * theta)) / math.factorial(int(round(N)) - 1 if N > 1 else 1)

# --------------------------------------------------------
# 6️⃣ Fit N using scipy curve_fit
# --------------------------------------------------------
# Only use data where E_theta > 0 to avoid log noise
mask = E_theta > 0
theta_fit = theta[mask]
E_fit = E_theta[mask]

try:
    popt, _ = curve_fit(lambda th, N: (N**N * th**(N - 1) * np.exp(-N * th)) / math.gamma(N),
                        theta_fit, E_fit, p0=[3], bounds=(1, 50))
    N_curve = popt[0]
except Exception as e:
    print(f"Curve-fit failed: {e}")
    N_curve = np.nan

# --------------------------------------------------------
# 7️⃣ Compute N via moment (variance) method
# --------------------------------------------------------
sigma_theta_sq = np.sum(((theta - 1)**2) * E_theta * np.mean(np.diff(theta))) / np.sum(E_theta * np.mean(np.diff(theta)))
N_moment = 1 / sigma_theta_sq

# --------------------------------------------------------
# 8️⃣ Print results
# --------------------------------------------------------
print(f"\nEstimated N (curve-fit)   = {N_curve:.3f}")
print(f"Estimated N (moment)      = {N_moment:.3f}")

# --------------------------------------------------------
# 9️⃣ Plot results
# --------------------------------------------------------
plt.figure(figsize=(7,5))
plt.plot(theta, E_theta, 'bo', label='CFD E(θ)', markersize=3)
if not np.isnan(N_curve):
    E_model = (N_curve**N_curve * theta**(N_curve - 1) * np.exp(-N_curve * theta)) / math.gamma(N_curve)
    plt.plot(theta, E_model, 'r-', label=f'TiS model (N={N_curve:.2f})', linewidth=1.8)
plt.xlabel("Dimensionless time θ = t/τ")
plt.ylabel("E(θ)")
plt.title("Tank-in-Series Fit to RTD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
