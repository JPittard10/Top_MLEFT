#!/usr/bin/env python3
"""
ML4EFT_train.py — Parametric Classifier for EFT Likelihood Ratio Estimation using ML4EFT
--------------------------------------------------------------------

Author: James Pittard @ Purdue CMS Top Group
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
# 1. Data utilities
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

def ML4EFT_generate_data(reweighter, var_names, wc_dim, N, M, theta1):
    """Yield (X, Y, theta) batches following the Brehmer procedure."""
    values = np.linspace(-2, 2, N)
    for v in values:
        th0 = np.zeros(wc_dim, dtype=float)
        th0[0] = v
        obs0 = reweighter.resample_observables(th0, max_events=M)
        obs1 = reweighter.resample_observables(theta1, max_events=M)
        X, Y = build_dataset(obs0, obs1, th0, theta1, var_names)
        yield X, Y, th0
