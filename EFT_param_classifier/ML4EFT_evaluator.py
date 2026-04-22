#!/usr/bin/env python3
"""
EFTReweighter for paramic classifier using ML4EFT
-------------

Author: James Pittard @ Purdue CMS Top Group
"""

import os
import numpy as np
import torch
import uproot
import awkward as ak
from tqdm import tqdm
import sys

def resample_observables(self, wc_point, max_events=None):
    """
    Resample observables according to EFT-weighted probabilities.
    Used for generating pseudo-data samples.

    Args:
        wc_point (list): EFT Wilson coefficients
        max_events (int, optional): limit number of events for testing

    Returns:
        dict[str, np.ndarray]: resampled observables
    """

    data = {k: v[mask] for k, v in self.final_observables.items()}
    weights = ak.to_numpy(self.get_final_weights(wc_point)[mask]).clip(min=0)

    # Optional event limit
    if max_events is not None:
        max_events = min(max_events, len(weights))
        idx = np.random.choice(len(weights), size=max_events, replace=False)
        weights = weights[idx]
        data = {k: v[idx] for k, v in data.items()}

    weights_sum = np.sum(weights)
    if weights_sum == 0:
        raise RuntimeError("All event weights are zero after masking.")
    weights /= weights_sum

    idx_sampled = np.random.choice(len(weights), size=len(weights), p=weights)
    sampled = {k: v[idx_sampled] for k, v in data.items()}

    print(f"[INFO] Resampled {len(weights)} events (step={self.step}) with EFT-weighted probabilities.")
    return sampled

def ML4EFT_resample_observables(some data needs to go here,wc_point, max_events = None):
    data = {k: v[mask] for k, v in self.final_observables.items()}

    print(f"[INFO] Resampled {len(weights)} events (step={self.step}) with EFT-weighted probabilities.")
    return sampled