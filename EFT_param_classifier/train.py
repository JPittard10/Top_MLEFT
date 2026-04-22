#!/usr/bin/env python3
"""
train.py — Parametric Classifier for EFT Likelihood Ratio Estimation
--------------------------------------------------------------------
Implements the "Calibrated Classifiers" method (Brehmer et al.)
using event samples resampled via EFTReweighter.

Adds:
  • Per-θ input-feature plots (EFT vs SM)
  • Feature-correlation matrix after training

Author: Santosh Bhandari @ Purdue CMS Top Group
"""

import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from evaluator import EFTReweighter  # your existing class


# ===============================================================
# 1. Model definition
# ===============================================================
class ParametricClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        layers = []
        hidden = 128
        for _ in range(3):
            layers += [
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                nn.Dropout(0.1),
            ]
            input_dim = hidden
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ===============================================================
# 2. Data utilities
# ===============================================================
def build_dataset(obs0, obs1, th0, th1, var_names):
    """Build combined dataset for (θ0 vs θ1)."""
    X0 = np.stack([np.asarray(obs0[k]) for k in var_names], axis=1)
    X1 = np.stack([np.asarray(obs1[k]) for k in var_names], axis=1)
    TH0 = np.tile(np.asarray(th0, float), (len(X0), 1))
    TH1 = np.tile(np.asarray(th1, float), (len(X1), 1))
    X = np.vstack([np.hstack([X0, TH0]), np.hstack([X1, TH1])])
    Y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])
    return X, Y


def generate_data(reweighter, var_names, wc_dim, N, M, theta1):
    """Yield (X, Y, theta) batches following the Brehmer procedure."""
    values = np.linspace(-2, 2, N)
    for v in values:
        th0 = np.zeros(wc_dim, dtype=float)
        th0[0] = v
        obs0 = reweighter.resample_observables(th0, max_events=M)
        obs1 = reweighter.resample_observables(theta1, max_events=M)
        X, Y = build_dataset(obs0, obs1, th0, theta1, var_names)
        yield X, Y, th0


# ===============================================================
# 3. Training
# ===============================================================
def train_model(X, Y, device, input_dim, epochs=5):
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = ParametricClassifier(input_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    def make_loader(X, Y, shuffle=True):
        return DataLoader(
            TensorDataset(torch.tensor(X).float(), torch.tensor(Y).float().unsqueeze(1)),
            batch_size=1024,
            shuffle=shuffle,
        )

    train_loader = make_loader(X_train, Y_train)
    val_loader = make_loader(X_val, Y_val, shuffle=False)

    best_loss, best_state = float("inf"), None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_train += loss.item() * len(xb)
        train_loss = total_train / len(train_loader.dataset)

        model.eval()
        total_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                total_val += loss_fn(pred, yb).item() * len(xb)
        val_loss = total_val / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss, best_state = val_loss, model.state_dict()

    model.load_state_dict(best_state)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training History")
    plt.tight_layout()
    plt.show()

    return model, scaler, (X_test, Y_test)


# ===============================================================
# 4. Calibration
# ===============================================================
class CalibratedParametric:
    def __init__(self, model, scaler, device):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.iso_by_theta = {}

    @torch.no_grad()
    def _predict_s(self, X):
        Xs = self.scaler.transform(X)
        tX = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        s = self.model(tX).cpu().numpy().ravel()
        return np.clip(s, 1e-6, 1 - 1e-6)

    def fit_isotonic_for_theta(self, X_calib, y_calib, theta_eval):
        s = self._predict_s(X_calib)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(s, y_calib.astype(float))
        self.iso_by_theta[tuple(np.asarray(theta_eval, float))] = iso

    def r_ratio(self, X, theta_eval):
        """
        Compute calibrated likelihood ratio r(x|θ0, θ1) = (1 - s_cal) / s_cal
        with numerical stabilization and safety clipping.
        """
        key = tuple(np.asarray(theta_eval, float))
        if key not in self.iso_by_theta:
            raise KeyError(f"No calibration found for theta={theta_eval}")
    
        # raw classifier scores (0–1)
        s_raw = self._predict_s(X)
    
        # calibrated isotonic scores (can saturate near 0 or 1)
        s_cal = self.iso_by_theta[key].transform(s_raw)
    
        # numerical stabilization
        eps = 1e-8
        s_cal = np.clip(s_cal, eps, 1 - eps)
    
        # compute ratio
        r = (1.0 - s_cal) / s_cal
    
        # also clip final ratio to avoid inf in log(r)
        r = np.clip(r, eps, 1e8)
    
        return r



# ===============================================================
# 5. Plotting utilities (after training)
# ===============================================================
def plot_features_per_theta(thetas_all, X_batches, Y_batches, VARS, outdir="artifacts/plots"):
    os.makedirs(outdir, exist_ok=True)
    print(f" Plotting per-θ feature distributions → {outdir}")

    for (X, Y, th0) in zip(X_batches, Y_batches, thetas_all):
        theta_label = f"{th0[0]:+.2f}"
        subdir = os.path.join(outdir, f"theta_{theta_label}")
        os.makedirs(subdir, exist_ok=True)

        # Plot each observable for SM vs EFT
        for i, var in enumerate(VARS):
            vals_sm = X[Y == 1, i]   # SM (θ₁)
            vals_eft = X[Y == 0, i]  # EFT (θ₀)

            # Binning
            if var == "gen_ttbar_mass":
                bin_edges = np.linspace(350, 1500, 50)
            else:
                combined = np.concatenate([vals_sm, vals_eft])
                low, high = np.min(combined), np.max(combined)
                if np.isfinite(low) and np.isfinite(high) and low < high:
                    bin_edges = np.linspace(low, high, 50)
                else:
                    bin_edges = 50

            plt.figure(figsize=(6, 4))
            plt.hist(vals_sm, bins=bin_edges, density=True, histtype='step', label="SM (θ₁)", color='gray', linewidth=2)
            plt.hist(vals_eft, bins=bin_edges, density=True, histtype='step', label=f"EFT θ₀={theta_label}", color='C0', linewidth=2)
            plt.xlabel(var)
            plt.ylabel("Normalized entries")
            plt.title(f"{var} — θ₀={theta_label}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            if var == "gen_ttbar_mass":
                plt.yscale('log')
                plt.xlim(300, 1500)
            plt.tight_layout()
            plt.savefig(os.path.join(subdir, f"{var}.png"))
            plt.close()

        # Correlation matrix for this θ₀ (only the observable columns, not θ)
        df = pd.DataFrame(X[:, :len(VARS)], columns=VARS)
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(), cmap="coolwarm", vmin=-1, vmax=1, annot=False)
        plt.title(f"Feature Correlation Matrix — θ₀={theta_label}")
        plt.tight_layout()
        plt.savefig(os.path.join(subdir, "corr_matrix.png"))
        plt.close()


# ===============================================================
# 6. Main training entry
# ===============================================================
if __name__ == "__main__":
    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wc_dim = 16

    # >>> Replace with your actual feature list
    VARS = ["gen_ll_cHel", "gen_ttbar_mass"]

    # Scan and per-theta sample sizes
    N, M = 3, 1000000
    theta1 = [0.0] * wc_dim
    step = 0  # 0 = GEN, 8 = RECO

    # Paths (update as needed)
    mass_regions = ["0to700", "700to900", "900toInf"]
    cross_sections = {"0to700": 65.09, "700to900": 8.295, "900toInf": 14.03}
    directory = "/eos/purdue/store/user/lingqian/fullrun2_eft_minitrees"
    struct_dir = "/depot/cms/top/he614/notebooks/EFT_FullRun2/"
    eras = ["2016preVFP"]
    channels = ["ee", "emu", "mumu"]

    # -------------------------
    print(" Loading EFT samples ...")
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

    # -------------------------
    print(" Building combined dataset ...")
    X_batches, Y_batches, thetas_all = [], [], []
    for X, Y, th0 in generate_data(reweighter, VARS, wc_dim, N, M, theta1):
        X_batches.append(X)
        Y_batches.append(Y)
        thetas_all.append(th0)
    X_all = np.vstack(X_batches)
    Y_all = np.concatenate(Y_batches)
    print(f"Total samples: {len(Y_all):,}")

    # -------------------------
    print(" Training classifier ...")
    model, scaler, (X_test, Y_test) = train_model(X_all, Y_all, device, X_all.shape[1])

    # -------------------------
    print(" Calibrating ...")
    calibrator = CalibratedParametric(model, scaler, device)
    for Xc, Yc, th0 in generate_data(reweighter, VARS, wc_dim, N=3, M=3000, theta1=theta1):
        calibrator.fit_isotonic_for_theta(Xc, Yc, th0)

    # -------------------------
    print(" Saving artifacts ...")
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model.pt")
    joblib.dump(scaler, "artifacts/scaler.pkl")
    joblib.dump(calibrator.iso_by_theta, "artifacts/calibrations.pkl")

    # -------------------------
    print(" Plotting per-θ features & correlations ...")
    plot_features_per_theta(thetas_all, X_batches, Y_batches, VARS)

    print(" Training and plotting complete.")
