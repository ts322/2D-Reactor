import numpy as np
import matplotlib.pyplot as plt
import os

def compute_rtd(time, signal):
    t = np.array(time)
    s = np.array(signal)

    # Detect average dt (even if irregular spacing)
    dt = np.mean(np.diff(t))

    # Normalise signal (allowing negative values)
    integral = np.sum(s * dt)
    Et = s / integral

    # Mean residence time τ
    tau = np.sum(t * s * dt) / integral

    # Dimensionless variables
    theta = t / tau
    E_theta = tau * Et

    return Et, tau, theta, E_theta

# -----------------------------
# USER INPUT
case_dir = "/home/ts322/ResearchProject/4th-Year-Research-Project/2D_Reactor/Results_2025-10-01_12-30"
data_file = os.path.join(case_dir, "postProcessing", "patchAverage_massfraction", "0", "s")
# -----------------------------

# Load all time points (skip header lines starting with #)
data = np.loadtxt(data_file, comments="#")
time = data[:,0]
signal = data[:,1]

# Compute RTD
Et, tau, theta, E_theta = compute_rtd(time, signal)

print(f"Mean residence time τ = {tau:.6f} s")
print(f"Simulation time range: {time[0]:.3f} → {time[-1]:.3f} s, {len(time)} samples")
print(f"Total signal integral (∫s(t)dt) = {np.sum(signal*np.mean(np.diff(time))):.6e}")

# Plot E(t)
plt.figure(figsize=(8,4))
plt.plot(time, Et, 'k-', label="E(t)")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Time t [s]")
plt.ylabel("E(t)")
plt.title("Exit Age Distribution E(t)")
plt.legend()
plt.grid(True)

# Plot E(θ)
plt.figure(figsize=(8,4))
plt.plot(theta, E_theta, 'r-', label="E(θ)")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Dimensionless time θ")
plt.ylabel("E(θ)")
plt.title("Dimensionless Exit Age Distribution E(θ)")
plt.legend()
plt.grid(True)

plt.show()
