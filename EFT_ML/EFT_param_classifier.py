#/depot/cms/kernels//python3/bin/python

import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import joblib
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 2112
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# 1.  MODEL  —  Parametric Classifier
# =============================================================================

class ParametricClassifier(nn.Module):
    """
    Multi-layer perceptron.
    Input dimension = len(VARS) + wc_dim.
    Output = sigmoid scalar ∈ (0,1)  →  interpreted as s(x,θ).
    """
    def __init__(self, input_dim: int, hidden: int = 128, n_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                nn.Dropout(0.1)
            ]
            in_dim = hidden
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
# =============================================================================
# 2.  DATA GENERATION
# =============================================================================
# Algorithm from thesis section 7.1:
#   • Choose N random θ₀ values uniformly in [θ_min, θ_max]
#   • For each θ₀:
#       – Draw M events from p(·|θ₀)  → label 0,  append θ₀ as extra features
#       – Draw M events from p(·|θ₁)  → label 1,  append θ₀ as extra features
#                                        ^^^^ SAME θ₀ appended to both classes
#   • Total dataset size: 2 · N · M

def generate_data(reweighter, var_names: list, wc_dim: int, N: int,
                  M: int, theta1: list, 
                  wc_scan_index: int = 0,
                  wc_range: tuple = (-2.0, 2.0)):
    
    """
    Yields (X, Y, theta0_used) for each of the N θ₀ scan points.

    X shape: (2*M, len(var_names) + wc_dim)
    Y shape: (2*M,)  — 0 = EFT sample, 1 = SM sample
    theta0_used: np.ndarray of shape (wc_dim,)

    NOTE: both EFT and SM event rows carry the *same* θ₀ as input features.
    This is correct — the network learns to separate the *distributions*
    given a hypothesis, not to memorise which class θ₀ belongs to.
    """
    theta_values = np.linspace(wc_range[0], wc_range[1], N)

    for v in theta_values:
        th0 = np.zeros(wc_dim, dtype=float)
        th0[wc_scan_index] = v

        # Draw events from p(x|θ₀)
        obs0 = reweighter.resample_observables(th0.tolist(),
                                            max_events = M)
        # Draw events from p(x|θ₁=SM)
        obs1 = reweighter.resample_observables(theta1,
                                            max_events = M)
        
        X0 = np.stack([np.asarray(obs0[k]) for k in var_names], axis = 1)
        X1 = np.stack([np.asarray(obs1[k]) for k in var_names], axis = 1)

        TH = np.tile(th0,(len(X0) + len(X1), 1))
        X = np.vstack([X0, X1]).astype(np.float32)
        X_full = np.hstack([X,TH]).astype(np.float32)

        Y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))]).astype(np.float32)

        w0 = np.asarray(reweighter.get_final_weights(th0.tolist())[:len(X0)])
        w1 = np.asarray(reweighter.get_final_weights(theta1)[:len(X1)])
        weights = np.concatenate([w0,w1]).astype(np.float32)

        yield X_full, Y, th0, weights

# =============================================================================
# 3.  TRAINING
# =============================================================================

def train_model(X: np.ndarray, Y: np.ndarray,
                device,
                epochs: int = 30,
                batch_size: int = 1024,
                lr: float = 1e-3,
                val_frac: float = 0.15,
                holdout_frac: int = 0.10):
    """
    Train the parametric classifier.

    Returns
    -------
    model   : best-checkpoint keras model
    scaler  : fitted StandardScaler (fit on training split only)
    splits  : dict with 'test' and 'holdout' (X, Y, W) arrays for diagnostics
    history : dict with train/val loss lists
    """
    # ── splits ───────────────────────────────────────────────────────────────
    idx = np.arange(len(Y))
    idx_dev, idx_hold = train_test_split(idx, test_size = holdout_frac, random_state = SEED)
    idx_tr, idx_val = train_test_split(idx_dev, test_size = val_frac, random_state = SEED)

    scaler = StandardScaler().fit(X[idx_tr])
    Xs = scaler.transform(X)

    def _loader(idxs, shuffle = True):
        Xt = torch.tensor(Xs[idxs]).float()
        Yt = torch.tensor(Y[idxs]).float().unsqueeze(1)
        return DataLoader(TensorDataset(Xt, Yt),
                          batch_size = batch_size, shuffle = shuffle)
    tr_loader  = _loader(idx_tr)
    val_loader = _loader(idx_val, shuffle = False)

    model     = ParametricClassifier(X.shape[1]).to(device)
    opt       = optim.Adam(model.parameters(), lr = lr)
    loss_fn   = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience = 3, factor = 0.5
    )

    best_val, best_state = float("inf"), None
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            t_loss += loss.item() * len(xb) # why this * len(xb)?
        t_loss /= len(idx_tr)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader: 
                xb, yb = xb.to(device), yb.to(device)
                v_loss += loss_fn(model(xb), yb).item() * len(xb) #### SAME HERE
        v_loss /= len(idx_val)

        history["train"].append(t_loss)
        history["val"].append(v_loss)
        scheduler.step(v_loss)
        print(f"Epoch {epoch+1:3d}/{epochs},  train = {t_loss:.4f},  val = {v_loss:.4f}")

        if v_loss < best_val: 
            best_val, best_state = v_loss, {k: v.clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"\n Best val loss: {best_val:.4f}")

    splits = {
        'test': (Xs[idx_val], Y[idx_val]),
        'holdout': (Xs[idx_hold], Y[idx_hold]),
    }

    return model, scaler, splits, history


def plot_training_history(history: dict, outdir: str = "artifacts/plots"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(7,4))
    plt.plot(history['train'], label = "train ", lw = 2)
    plt.plot(history['val'], label = "val ", lw = 2)
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/trianing_history.png", dpi = 150)
    plt.close()
    print(f"Saved {outdir}/trianing_history.pdf")

# =============================================================================
# 4.  CALIBRATION  (section 7.1 of thesis — weighted isotonic regression)
# =============================================================================
# KEY FIX vs your original code:
#
# Your original CalibratedParametric stored one IsotonicRegression object
# *per θ point* (keyed by exact tuple), and then failed at new θ values.
#
# The thesis says: "one can use a SINGLE set for all θ and use the weights
# as weights in the isotonic regression."
#
# Correct procedure:
#   1. Pool events from *all* θ calibration points.
#   2. Compute raw classifier score s for each event.
#   3. Fit ONE isotonic regression of  (s → true label)
#      with the EFT MC weights as sample_weight.
#   4. At inference: apply this single calibration to any θ.


class CalibratedParametric:
    """
    Single weighted isotonic calibration (as prescribed by thesis section 7.1).
    Works for ANY θ at inference — no per-θ lookup needed.
    """

    def __init__(self, model: ParametricClassifier, scaler: StandardScaler, device):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.iso    = None  #Fitted IsotonicRegression

    @torch.no_grad()
    def _raw_score(self, X: np.ndarray) -> np.ndarray:
        """Raw sigmoid output s(x,θ) ∈ (0,1)."""
        Xs = self.scaler.transform(X)
        t  = torch.tensor(Xs, dtype = torch.float32, device = self.device)
        s  = self.model(t).cpu().numpy().ravel()
        return np.clip(s, 1e-7, 1 - 1e-7)

    def fit_calibration(self, X_pool: np.ndarray, Y_pool: np.ndarray, W_pool: np.ndarray):
        """
        Fit a single isotonic regression on pooled calibration data.

        Parameters
        ----------
        X_pool : (N, input_dim)  — features + θ appended
        Y_pool : (N,)            — binary labels (0=EFT, 1=SM)
        W_pool : (N,)            — EFT MC event weights (used as sample_weight)
        """
        s_raw = self._raw_score(X_pool)
        W_pool_norm = W_pool / (W_pool.sum() + 1e-12)
        self.iso = IsotonicRegression(out_of_bounds = "clip", increasing = True)
        self.iso.fit(s_raw, Y_pool.astype(float), sample_weight = W_pool_norm)
        print(f"Isotonic Calibration fitted on {len(Y_pool):,} events.")

    def calibrated_score(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated s_cal(x,θ) ∈ (0,1)."""
        if self.iso is None:
            raise RuntimeError("Call fit_calibration() first.")
        s_raw = self._raw_score(X)
        s_cal = self.iso.transform(s_raw)
        return np.clip(s_cal,1e-8, 1 - 1e-8) #Why is it 1e-8 for calibrated score and 1e-7 for _raw_score !!!!!!!!!!!!!!!!!!!!

    def log_likelihood_ratio(self, X: np.ndarray) -> np.ndarray:
        """
        log r(x|θ₀,θ₁) = log[(1-s_cal)/s_cal]

        Positive values → x is more likely under θ₀ (EFT).
        Negative values → x is more likely under θ₁ (SM).
        """
        s = self.calibrated_score(X)
        return np.log((1.0 - s) / s)

# =============================================================================
# 5.  CALIBRATION DIAGNOSTICS
# =============================================================================

def calibration_closure_plot(calibrator: CalibratedParametric,
                             X_holdout: np.ndarray,
                             Y_holdout: np.ndarray,
                             process_name: str = "",
                             nbins: int = 40,
                             outdir: str = "artificats/plots"):
    
    """
    Calibration closure: ŝ_NN vs N_EFT/(N_EFT+N_SM) in bins of ŝ_NN.
    Should lie on the y=x diagonal if calibration is correct.
    """

    os.makedirs(outdir, exist_ok = True)
    s_cal = calibrator.calibrated_score(X_holdout)
    bins = np.linspace(0,1, nbins + 1)
    mc_frac, nn_means = [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (s_cal >= lo) & (s_cal < hi)
        if m.sum() < 2:
            continue
        n_eft = (Y_holdout[m] == 0).sum()
        n_sm  = (Y_holdout[m] == 1).sum()
        mc_frac.append(n_sm / (n_sm + n_eft + 1e-12))
        nn_means.append(s_cal[m].mean())

    fig, ax = plt.subplots(figsize = (6,5))
    ax.plot([0,1], [0,1], 'k--', lw = 1, label = "Ideal (y = x)")
    ax.scatter(nn_means, mc_frac, s = 18, label = f"{process_name}")
    ax.set_xlabel("Mean calibrated ŝ in bin", fontsize = 12)
    ax.set_ylabel("MC fraction N_SM / (N_SM + N_EFT)", fontsize = 12)
    ax.set_title(f"Calibration Closure {process_name}", fontsize = 12)
    ax.legend()
    plt.tight_layout()
    fname = f"{outdir}/calibration_closure{'_'+process_name if process_name else ''}.pdf"
    plt.savefig(fname, dpi = 150)
    plt.close()
    print(f"Saved {fname}")


def roc_curve_plot(calibrator: CalibratedParametric,
                   X_test: np.ndarray,
                   Y_test: np.ndarray,
                   outdir: str = "artifacts/plots"):
    """ROC curve of the calibrated classifier."""
    from sklearn.metrics import roc_curve, roc_auc_score
    os.makedirs(outdir, exist_ok = True)
    s = calibrator.calibrated_score(X_test)
    fpr, tpr, _  = roc_curve(Y_test, s)
    auc = roc_auc_score(Y_test, s)

    fig, ax = plt.subplots(figsize = (6,5))
    ax.plot(fpr, tpr, lw = 2, label = f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1], 'k--', lw = 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    fname = f"{outdir}/roc_curve.pdf"
    plt.savefig(fname, dpi = 150)
    plt.close()
    print(f"Saved {fname}")

# =============================================================================
# 6.  TEST STATISTIC
# =============================================================================
# Profile likelihood ratio test statistic (Asimov / observed):
#
#   t(θ) = -2 Σᵢ log r(xᵢ | θ, θ_SM)
#         = -2 Σᵢ log [(1 - ŝᵢ(θ)) / ŝᵢ(θ)]
#
# For a 1D scan over one Wilson coefficient:
#   • Compute t_asimov(θ) using SM-generated events  → expected sensitivity
#   • Compute t_obs(θ) using observed / pseudo-data events

def build_input_at_theta(X_obs: np.ndarray, theta: np.ndarray, n_vars: int) -> np.ndarray:
    """
    Replace the θ columns in X_obs with a new θ value.

    Parameters
    ----------
    X_obs  : (N, n_vars + wc_dim) — feature matrix with old θ appended
    theta  : (wc_dim,)            — new Wilson coefficient point
    n_vars : int                  — number of kinematic features
    """
    X_new = X_obs.copy()
    X_new[:, n_vars:] = theta[np.newaxis, :]
    return X_new.astype(np.float32)


def compute_test_statistic(calibrator: CalibratedParametric,
                           X_events: np.ndarray,
                           theta: np.ndarray,
                           n_vars: int,
                           event_weights: np.ndarray = None) -> float:
    """
    Compute t(θ) = -2 Σᵢ wᵢ · log r(xᵢ|θ, θ_SM)
                 = -2 Σᵢ wᵢ · log[(1-ŝᵢ)/ŝᵢ]

    Parameters
    ----------
    X_events      : (N, n_vars + wc_dim)  events with their original θ columns
    theta         : (wc_dim,)             hypothesis to test
    n_vars        : number of kinematic features (not counting WC columns)
    event_weights : (N,) optional MC weights (use None for unweighted)
    """
    X_at_theta = build_input_at_theta(X_events, theta, n_vars)
    log_r      = calibrator.log_likelihood_ratio(X_at_theta)

    if event_weights is not None:
        w = event_weights / event_weights.sum()
        t = -2.0 * np.sum(w * log_r) * len(log_r)   # restore scale
    else:
        t = -2.0 * np.sum(log_r)                                            # Wouldnt there need to be weights?

    return float(t)


def scan_test_statistic(calibrator: CalibratedParametric,
                        X_sm_events: np.ndarray,
                        X_obs_events: np.ndarray,
                        theta_scan: np.ndarray,
                        wc_scan_index: int,
                        wc_dim: int,
                        n_vars: int,
                        w_sm: np.ndarray = None,
                        w_obs: np.ndarray = None) -> dict:
    """
    Scan t(θ) over a range of Wilson coefficient values.

    Parameters
    ----------
    X_sm_events  : SM events (for Asimov / expected)
    X_obs_events : observed / pseudo-data events
    theta_scan   : 1D array of WC values to test
    wc_scan_index: which WC index is being scanned

    Returns
    -------
    dict with keys 'theta', 't_asimov', 't_obs'
    """
    t_asimov, t_obs = [], []
    
    for v in tqdm(theta_scan, desc="Scanning θ"):
        theta = np.zeros(wc_dim)
        theta[wc_scan_index] = v

        t_asimov.append(compute_test_statistic(
            calibrator, X_sm_events,  theta, n_vars, w_sm))
        t_obs.append(compute_test_statistic(
            calibrator, X_obs_events, theta, n_vars, w_obs))
        
    return { 
        "theta": theta_scan,
        "t_asimov": np.array(t_asimov),
        "t_obs": np.array(t_obs),
    }

# =============================================================================
# 7.  LIMITS  (CLs method)
# =============================================================================
# Under Wilks' theorem, t(θ) is asymptotically χ²(1) distributed.
# The 95% CL exclusion region is where t(θ) > χ²(0.95, 1) = 3.84.
#
# We implement both:
#   a) Asymptotic (Asimov) expected limits from t_asimov
#   b) Observed limits from t_obs

CRITICAL_VALUE_95 = 3.84    # χ²(0.95, 1) — 95% CL one-sided
CRITICAL_VALUE_68 = 1.00    # χ²(0.68, 1) — 68% CL

def extract_limits(theta_scan: np.ndarray, t_values: np.ndarray,
                   critical_value: float = CRITICAL_VALUE_95) -> tuple:
    """
    Find the crossing points of t(θ) = critical_value.

    Returns (lower_limit, upper_limit).
    If only one crossing, returns (None, upper) or (lower, None).
    """
    crossings = []
    for i in range(len(t_values) - 1):
        if ((t_values[i] - critical_value) * (t_values[i+1] - critical_value)) < 0:
            # Linear Interpolation
            frac = (critical_value - t_values[i]) / (t_values[i+1] - t_values[i])
            crossing = theta_scan[i] + frac * (theta_scan[i+1] - theta_scan [i])
            crossings.append(crossing)

    if len(crossings) >= 2:
        return crossings[0], crossings[-1]
    elif len(crossings) == 1:
        if theta_scan[np.argmin(t_values)] < crossings[0]:
            return None, crossings[0]
        else:
            return crossings[0], None
    else:
        return None, None
    
def compute_all_limits(scan_results: dict,
                       wc_name: str = "θ") -> dict:
    """
    Compute 68% and 95% CL limits for expected (Asimov) and observed.

    Returns a dict with all limit values and best-fit θ.
    """
    theta  = scan_results["theta"]
    t_asim = scan_results["t_asimov"]
    t_obs  = scan_results["t_obs"]

    #Best Fit
    bf_asimov = theta[np.argmin(t_asim)]
    bf_obs    = theta[np.argmin(t_obs)]

    dt_asim = t_asim - np.min(t_asim)
    dt_obs  = t_obs - np.min(t_obs)

    results = {
        "wc_name": wc_name,
        "best_fit_asimov": bf_asimov,
        "best_fit_obs": bf_obs
    }

    for level, cv in [("95", CRITICAL_VALUE_95), ("68", CRITICAL_VALUE_68)]:
        lo_a, hi_a = extract_limits(theta, dt_asim, cv)
        lo_o, hi_o = extract_limits(theta, dt_obs, cv)
        
        results[f"expected_{level}_low"] = lo_a
        results[f"expected_{level}_high"] = hi_a
        results[f"observed_{level}_low"] = lo_o
        results[f"observed_{level}_high"] = hi_o

    return results

def print_limits(limits: dict):
    wc = limits["wc_name"]
    print(f"\n{'='*55}")
    print(f"Limits on {wc}")
    print(f"{'='*55}")
    print(f"Best fit (Asimov): {limits['best_fit_asimov']:.3f}")
    print(f"Best fit (Obs): {limits['best_fit_obs']:.3f}")
    print()
    for kind in ["expected", "observed"]:
        for level in ["95", "68"]:
            lo = limits[f"{kind}_{level}_low"]
            hi = limits[f"{kind}_{level}_high"]
            lo_s = f"{lo:.3f}" if lo is not None else "-∞"
            hi_s = f"{hi:.3f}" if hi is not None else "+∞"
            print(f" {kind.capitalize()} {level}% CL:  [{lo_s}, {hi_s}]")
    print()

# =============================================================================
# 8.  PLOTTING
# =============================================================================

def plot_test_statistic_scan(scan_results: dict,
                             limits: dict,
                             wc_name: str = "θ",
                             outdir: str = "artifacts/plots"):
    """
    Plot t(θ) scan with 68% / 95% CL lines, best-fit marker, and limit bands.
    """
    os.makedirs(outdir, exist_ok = True)
    theta = scan_results["theta"]
    t_asim = scan_results["t_asimov"]
    t_obs = scan_results["t_obs"]

    t_asim = t_asim - np.min(t_asim)
    t_obs  = t_obs - np.min(t_obs)

    fig, ax = plt.subplots(figsize = (8,5))

    ax.plot(theta, t_asim, 'k--', lw = 2, label = "Expected (Asimov)")
    ax.plot(theta, t_obs, 'b-', lw = 2, label = "Observed")

    ax.axhline(CRITICAL_VALUE_95, color='r', ls=':', lw=1.5, label="95% CL (3.84)")
    ax.axhline(CRITICAL_VALUE_68, color = 'g', ls = ':', lw = 1.5, label = "68% CL (1.00)")
    ax.axhline(0.0, color = 'black', ls = '-', lw = 0.8)

    #best-fit markers

    ax.axvline(limits["best_fit_obs"], color='b', ls='--', lw=1,
               label=f"Best fit: {limits['best_fit_obs']:.3f}")
    
    for side, key, col in [("lower", "observed_95_low", "b"),
                           ("upper", "observed_95_high", "b")]:
        v = limits[key]
        if v is not None:
            ax.axvline(v, color = col, ls = '-.', lw = 1.2)
    
    ax.set_xlabel(f"Wilson coefficient  {wc_name}", fontsize=13)
    ax.set_ylabel(r"$-2\,\Delta\log\mathcal{L}$", fontsize=13)
    ax.set_title(f"Profile Likelihood Scan — {wc_name}", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"{outdir}/test_stat_scan_{wc_name}.pdf"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved {fname}")
    

def plot_log_r_distribution(calibrator: CalibratedParametric,
                            X_sm: np.ndarray,
                            X_eft: np.ndarray,
                            theta_eft: np.ndarray,
                            n_vars: int,
                            wc_name: str = "θ",
                            wc_val: float = 0.0,
                            outdir: str = "artifacts/plots"):
    
    """
    Distribution of log r(x|θ_EFT, θ_SM) for SM and EFT events.
    A good classifier gives clearly separated distributions.
    """
    os.makedirs(outdir, exist_ok = True)
    
    X_sm_at = build_input_at_theta(X_sm, theta_eft, n_vars)
    X_eft_at = build_input_at_theta(X_eft, theta_eft, n_vars)

    logr_sm = calibrator.log_likelihood_ratio(X_sm_at)
    logr_eft = calibrator.log_likelihood_ratio(X_eft_at)

    # Remove non-finite values of logr
    logr_sm = logr_sm[np.isfinite(logr_sm)]
    logr_eft = logr_eft[np.isfinite(logr_eft)]

    lo = min(np.percentile(logr_sm, 1), np.percentile(logr_eft, 1))
    hi = max(np.percentile(logr_sm, 99), np.percentile(logr_eft, 99))
    bins = np.linspace(lo, hi, 70) #                                why 70??????????????????????????????????????????????

    fig, ax = plt.subplots(figsize = (8,5))
    ax.hist(logr_sm, bins = bins, histtype = 'step', lw = 2,
            density = True, color = 'gray', label = r"SM ($\theta_1$)")

    ax.hist(logr_eft, bins = bins, histtype = 'step', lw = 2,
            density = True, color = 'C0',
            label = rf"EFT ({wc_name} = {wc_val:+.1f})")
    
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlabel(r"$\log\,r(x|\theta_0,\theta_1)$", fontsize=13)
    ax.set_ylabel("Normalized entries", fontsize=12)
    ax.set_title("Log-Likelihood Ratio Distribution", fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"{outdir}/logr_distribution_{wc_name}.pdf"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved {fname}")

def plot_2d_limit_contour(calibrator: CalibratedParametric,
                          X_sm: np.ndarray,
                          n_vars: int, 
                          wc_dim: int, 
                          wc_indices: tuple = (0,1),
                          wc_names: tuple = ("θ₀", "θ₁"),
                          scan_range: tuple = (-3.0, 3.0),
                          n_pts: int = 20,
                          outdir: str = "artifacts/plots"):
    """
    2D contour plot of t(θ) for two Wilson coefficients.
    Useful for showing the shape of the exclusion region.
    """
    os.makedirs(outdir, exist_ok = True)
    vals = np.linspace(scan_range[0], scan_range[1], n_pts)
    T    = np.zeros((n_pts, n_pts))
    for i, v0 in enumerate(tqdm(vals, desc = "2D scan axis 0")):
        for j, v1 in enumerate(vals):
            theta = np.zeros(wc_dim)
            theta[wc_indices[0]] = v0 
            theta[wc_indices[1]] = v1
            T[i,j] = compute_test_statistic(calibrator, X_sm, theta, n_vars)

    fig, ax = plt.subplots(figsize = (6,5))
    cf = ax.contourf(vals, vals, T.T, levels = np.linspace(0, T.max(), 30), cmap = 'Blues') # Why 30 for linspace ??????????????????????
    ax.contour(vals, vals, T.T, levels = [CRITICAL_VALUE_68, CRITICAL_VALUE_95],
               colors = ['green', 'red'], linewidths = 2)
    from matplotlib.lines import Line2D
    ax.legend(handles = [
        Line2D([0],[0], color = 'green', lw = 2, label = "68% CL"),
        Line2D([0],[0], color = 'red', lw = 2, label = "95% CL")
    ])
    plt.colorbar(cf, ax=ax, label=r"$-2\Delta\log\mathcal{L}$")
    ax.set_xlabel(wc_names[0], fontsize=12)
    ax.set_ylabel(wc_names[1], fontsize=12)
    ax.set_title("2D Exclusion Contour", fontsize=13)
    plt.tight_layout()
    fname = f"{outdir}/2d_contour_{wc_names[0]}_{wc_names[1]}.pdf"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved {fname}")

# =============================================================================
# 9.  ARTIFACT I/O
# =============================================================================

def save_artifacts(model, scaler, calibrator,
                   outdir: str = "artifacts"):
    os.makedirs(outdir, exist_ok = True)
    torch.save(model.state_dict(), f"{outdir}/model.pt")
    joblib.dump(scaler, f"{outdir}/scaler.pkl")
    joblib.dump(calibrator.iso, f"{outdir}/isotonic.pkl")
    print(f"Artifacts saved to {outdir}/")

def load_artifacts(input_dim: int, device,
                   outdir: str = "artifacts"):
    model = ParametricClassifier(input_dim).to(device)
    model.load_state_dict(torch.load(f"{outdir}/model.pt", map_location = device))
    model.eval()
    scaler = joblib.load(f"{outdir}/scaler.pkl")
    iso = joblib.load(f"{outdir}/isotonic.pkl")

    calibrator = CalibratedParametric(model, scaler, device)
    calibrator.iso = iso
    return model, scaler, calibrator

# =============================================================================
# 10. MAIN  —  Full End-to-End Pipeline
# =============================================================================

if __name__ == "__main__":
    from evaluator import EFTReweighter #existing class

    #-- Config --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wc_dim = 16
    theta1 = [0.0] * wc_dim # SM Hypothesis
    
    VARS = ["gen_ll_cHel", "gen_ttbar_mass", "gen_c_kk", "gen_c_rr", "gen_c_nn"] #More can be added
    n_vars = len(VARS)

    WC_NAMES = ['ctGRe', 'ctGIm', 'cQj18', 'cQj38', 'cQj11', 'cQj31',
                'ctu8',  'ctd8',  'ctj8',  'cQu8',  'cQd8',  'ctu1',
                'ctd1',  'ctj1',  'cQu1',  'cQd1']

    # WC to scan (index 0 = ctGRe)
    SCAN_WC_IDX  = 0
    SCAN_WC_NAME = WC_NAMES[SCAN_WC_IDX]
    SCAN_RANGE   = (-10.0,10.0)
    SCAN_N_PTS   = 81

    # Training hyperparameters
    N_THETA_TRAIN = 7      #  Number of theta points for thraining
    M_EVENTS      = 80000   #  events per theta class
    EPOCHS        = 20


    # Data Path --------------------------------------------------------------
    mass_regions    = ["0to700", "700to900", "900toInf"]
    cross_sections  = {"0to700": 65.09, "700to900": 8.295, "900toInf": 14.03}
    directory       = "/eos/purdue/store/user/lingqian/fullrun2_eft_minitrees"
    struct_dir      = "/depot/cms/top/he614/notebooks/EFT_FullRun2/"
    eras            = ["2016preVFP"]
    channels        = ["ee", "emu", "mumu"]

    USE_SAVED = False

    # Reweighter -------------------------------------------------------------
    print("\n[0] Loading EFT reweighter ...")
    rw = EFTReweighter(
        directory_path = directory, eras = eras, channels = channels,
        mass_regions = mass_regions, cross_sections = cross_sections,
        struct_const_dir = struct_dir, step = 0
    )
    rw.load_structure_constants()
    rw.load_observables()

    # Training data ----------------------------------------------------------
    if not USE_SAVED:
        print("\n[1] Building training dataset ...")
        X_batches, Y_batches, W_batches = [], [], []
        for X, Y, th0, W in generate_data(
            rw, VARS, wc_dim, 
            N = N_THETA_TRAIN, M = M_EVENTS,
            theta1 = theta1,
            wc_scan_index = SCAN_WC_IDX,
            wc_range = SCAN_RANGE):

            X_batches.append(X)
            Y_batches.append(Y)
            W_batches.append(W)

        X_all = np.vstack(X_batches)
        Y_all = np.concatenate(Y_batches)
        W_all = np.concatenate(W_batches)

        print(f" Total traning events: {len(Y_all):,}")

        # Train  -----------------------------------------------------------------
        print("\n[2] Training parametric classifier ...")
        model, scaler, splits, history = train_model(
            X_all, Y_all, device, epochs = EPOCHS)
        
        plot_training_history(history)

        # Calibration ------------------------------------------------------------
            # Use a held-out calibration set (separate from training if possible)
            # Here we reuse part of the training data for illustration.
            # In production: generate a fresh set of events for calibration.
        print("\n[3] Fitting weighted isotonic calibration ...")
        calibrator = CalibratedParametric(model, scaler, device)
        calibrator.fit_calibration(X_all, Y_all, W_all)

        # Save -------------------------------------------------------------------
        save_artifacts(model, scaler, calibrator)
    
    else: 
        print("\n [1-3] Loading saved artifacts ...")
        model, scaler, calibrator = load_artifacts(
            input_dim = n_vars + wc_dim, device = device)
        
        splits = None

    # Diagnostics ----------------------------------------------------------------
    print("\n[4]Diagnostics ...")
    if splits is not None:
        X_test, Y_test = splits["test"]
        calibration_closure_plot(calibrator, X_test, Y_test,
                                 process_name=SCAN_WC_NAME)
        roc_curve_plot(calibrator, X_test, Y_test)
    
    # Generate SM and EFT samples for test statistic _____________________________
    print("\n [5] Generating SM and EFT evaluation samples ...")
    sm_obs  = rw.resample_observables(theta1, max_events=200_000)
    X_sm    = np.stack([np.asarray(sm_obs[k]) for k in VARS], axis = 1)
    TH_sm   = np.tile(np.zeros(wc_dim), (len(X_sm), 1))
    X_sm    = np.hstack([X_sm, TH_sm]).astype(np.float32)

    theta_eft_demo = np.zeros(wc_dim)
    theta_eft_demo[SCAN_WC_IDX] = 2.0
    eft_obs = rw.resample_observables(theta_eft_demo.tolist(), max_events = 200_000)
    X_eft   = np.stack([np.asarray(eft_obs[k]) for k in VARS], axis = 1)
    TH_eft  = np.tile(theta_eft_demo, (len(X_eft), 1))
    X_eft   = np.hstack([X_eft, TH_eft]).astype(np.float32)

    plot_log_r_distribution(
        calibrator, X_sm, X_eft,
        theta_eft=theta_eft_demo,
        n_vars=n_vars,
        wc_name=SCAN_WC_NAME,
        wc_val=theta_eft_demo[SCAN_WC_IDX])
    
    # Test statistic scan --------------------------------------------------------
    print(f"\n[6] Scanning test Statistic over {SCAN_WC_NAME} ...")
    theta_scan = np.linspace(SCAN_RANGE[0], SCAN_RANGE[1], SCAN_N_PTS)

    scan_results = scan_test_statistic(
        calibrator,
        X_sm_events=X_sm,    # Asimov (expected): use SM events
        X_obs_events=X_sm,   # Replace with real observed data when unblinded
        theta_scan=theta_scan,
        wc_scan_index=SCAN_WC_IDX,
        wc_dim=wc_dim,
        n_vars=n_vars,
    )

    # Extract limits -------------------------------------------------------------
    print("\n[7] Computing limits ...")
    limits = compute_all_limits(scan_results, wc_name=SCAN_WC_NAME)
    print_limits(limits)

    # plots -----------------------------------------------------------------------
    print("\n[8] Making limit plots ...")
    plot_test_statistic_scan(scan_results, limits, wc_name=SCAN_WC_NAME)

    plot_2d_limit_contour(
         calibrator, X_sm, n_vars, wc_dim,
         wc_indices=(0, 8),
         wc_names=(WC_NAMES[0], WC_NAMES[8]))
    
    print("\n Finished script, All outputs in artifacts/")