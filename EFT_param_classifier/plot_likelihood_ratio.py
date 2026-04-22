#!/usr/bin/env python3
"""
plot_likelihood_ratio.py
------------------------
Loads trained parametric classifier, calibration maps, and StandardScaler.
Computes calibrated likelihood ratios r(x|θ₀,θ₁) for a chosen EFT point vs SM.

Produces:  Histogram of log(r) for SM and EFT samples.

Author: Santosh Bhandari @ Purdue CMS Top Group
"""

import os
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from train import ParametricClassifier, CalibratedParametric
from evaluator import EFTReweighter


# ===============================================================
# 1. Configuration
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wc_dim = 16
VARS = ["gen_ll_cHel", "gen_ttbar_mass"]  # same set as used in training
step = 0  # 0 = GEN, 8 = RECO
M = 600000  # number of events to evaluate

# EFT parameter points
theta1 = [0.0] * wc_dim  # SM hypothesis
theta0 = [0.0] * wc_dim
theta0[0] = 2.0          # example EFT hypothesis: ctGRe = +2

# Input paths
mass_regions = ["0to700", "700to900", "900toInf"]
cross_sections = {"0to700": 65.09, "700to900": 8.295, "900toInf": 14.03}
directory = "/eos/purdue/store/user/lingqian/fullrun2_eft_minitrees"
struct_dir = "/depot/cms/top/he614/notebooks/EFT_FullRun2/"
eras = ["2016preVFP"]
channels = ["ee", "emu", "mumu"]


# ===============================================================
# 2. Load trained artifacts
# ===============================================================
print(" Loading trained artifacts ...")
model = ParametricClassifier(len(VARS) + wc_dim)
model.load_state_dict(torch.load("artifacts/model.pt", map_location=device))
model.eval()

scaler = joblib.load("artifacts/scaler.pkl")
iso_by_theta = joblib.load("artifacts/calibrations.pkl")

calibrator = CalibratedParametric(model, scaler, device)
calibrator.iso_by_theta = iso_by_theta  # plug-in calibrated maps


# ===============================================================
# 3. Load reweighter and generate samples
# ===============================================================
print(" Loading EFT reweighter ...")
reweighter = EFTReweighter(
    directory_path=directory,
    eras=eras,
    channels=channels,
    mass_regions=mass_regions,
    cross_sections=cross_sections,
    struct_const_dir=struct_dir,
    step=step,
)
reweighter.load_structure_constants()
reweighter.load_observables()

def evaluate_observables(reweighter, wc_point, var_names, max_events=M):
    obs = reweighter.resample_observables(wc_point, max_events=max_events)
    return np.stack([np.asarray(obs[k]) for k in var_names], axis=1)

X0 = evaluate_observables(reweighter, theta0, VARS, max_events=M)
X1 = evaluate_observables(reweighter, theta1, VARS, max_events=M)

# Append theta inputs
TH0 = np.tile(np.asarray(theta0, float), (len(X0), 1))
TH1 = np.tile(np.asarray(theta1, float), (len(X1), 1))
X0_full = np.hstack([X0, TH0])
X1_full = np.hstack([X1, TH1])

# ===============================================================
# 4. Compute calibrated likelihood ratios
# ===============================================================
print("  Computing calibrated likelihood ratios ...")

r_eft = calibrator.r_ratio(X0_full, theta0)
r_sm  = calibrator.r_ratio(X1_full, theta0)

eps = 1e-8
r_eft = np.clip(r_eft, eps, 1e8)
r_sm  = np.clip(r_sm, eps, 1e8)

logr_eft = np.log(r_eft)
logr_sm  = np.log(r_sm)

# ===============================================================
# 4b. Diagnostics for non-finite values
# ===============================================================
def report_nonfinite(arr, name):
    mask_nan  = np.isnan(arr)
    mask_inf  = np.isinf(arr)
    n_nan = np.sum(mask_nan)
    n_inf = np.sum(mask_inf)
    if n_nan or n_inf:
        print(f"[WARN] {name} has {n_nan} NaN and {n_inf} inf values.")
        if n_nan:
            print(f"  NaN indices (first 10): {np.where(mask_nan)[0][:10]}")
        if n_inf:
            print(f"  Inf sample values (first 10): {arr[mask_inf][:10]}")
        finite_mask = np.isfinite(arr)
        print(f"  Finite range: min={np.nanmin(arr[finite_mask]):.3e}, "
              f"max={np.nanmax(arr[finite_mask]):.3e}")
    else:
        print(f"[INFO] {name} — all {len(arr)} entries finite "
              f"(range {np.min(arr):.3e} → {np.max(arr):.3e})")

report_nonfinite(logr_eft, "logr_eft")
report_nonfinite(logr_sm,  "logr_sm")

# Remove non-finite values before plotting
mask_finite_eft = np.isfinite(logr_eft)
mask_finite_sm  = np.isfinite(logr_sm)
removed_eft = len(logr_eft) - np.sum(mask_finite_eft)
removed_sm  = len(logr_sm)  - np.sum(mask_finite_sm)
if removed_eft or removed_sm:
    print(f"[INFO] Removed non-finite entries → EFT:{removed_eft}, SM:{removed_sm}")
logr_eft = logr_eft[mask_finite_eft]
logr_sm  = logr_sm[mask_finite_sm]


# ===============================================================
# 5. Plot log-likelihood ratio distributions
# ===============================================================
plt.figure(figsize=(8,5))
bins = np.linspace(-20, 20, 80)
plt.hist(logr_sm, bins=bins, histtype='step', linewidth=2, label="SM (θ₁)", density=True, color="gray")
plt.hist(logr_eft, bins=bins, histtype='step', linewidth=2, label=f"EFT (θ₀={theta0[0]:+.1f})", density=True, color="C0")
plt.xlabel("log r(x|θ₀, θ₁)")
plt.ylabel("Normalized entries")
plt.title("Likelihood-Ratio Distribution (SM vs EFT)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
os.makedirs("artifacts/plots", exist_ok=True)
plt.savefig("artifacts/plots/logr_distribution.png")
plt.show()

print(" Likelihood ratio plot saved to artifacts/plots/logr_distribution.png")
