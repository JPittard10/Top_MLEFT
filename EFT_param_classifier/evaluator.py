#!/usr/bin/env python3
"""
EFTReweighter
-------------
Loads EFT MiniAOD files and structure constants, computes deterministic event weights
for given EFT parameter points, and optionally resamples observables into pseudo-data.
Supports luminosity-weighted normalization per region and era.

Author: Jessie @ Purdue CMS Top Group
"""

import os
import numpy as np
import torch
import uproot
import awkward as ak
from tqdm import tqdm
import sys

# Import event-weight prediction module
sys.path.append('/depot/cms/top/bhanda25/Purdue_Analysis_EFT/Purdue_Analysis_EFT/EFT_minitree')
import Event_weight_prediction1


class EFTReweighter:
    def __init__(self, directory_path, eras, channels, mass_regions,
                 cross_sections, struct_const_dir, step):
        self.directory_path = directory_path
        self.cross_sections = cross_sections
        self.struct_const_dir = struct_const_dir
        self.step = step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------
        # Build ROOT file list: include split index 0–19 per mass region
        # ------------------------------------------------------------------
        self.file_paths = []
        for era in eras:
            for ch in channels:
                for reg in mass_regions:
                    for idx in range(20):
                        path = os.path.join(
                            directory_path,
                            f"spinCorrInput_{era}_August2025/Nominal/{ch}/"
                            f"{ch}_ttto2l2nu_jet_smeft_mtt_{reg}_{era}_{idx}.root"
                        )
                        if os.path.exists(path):
                            self.file_paths.append(path)
        if not self.file_paths:
            raise RuntimeError("❌ No ROOT files found — check your directory and naming pattern.")

        # ------------------------------------------------------------------
        # Build structure constant paths (aligned per ROOT file)
        # ------------------------------------------------------------------
        struct_step = "gen" if step == 0 else "reco"
        self.struct_paths = []
        for era in eras:
            for ch in channels:
                for reg in mass_regions:
                    for idx in range(20):
                        sc_path = os.path.join(
                            struct_const_dir,
                            f"saved_sc_{era}/Nominal/"
                            f"{ch}_ttto2l2nu_jet_smeft_mtt_{reg}_{era}_{idx}_struct_{struct_step}.npy"
                        )
                        if os.path.exists(sc_path):
                            self.struct_paths.append(sc_path)

        # ------------------------------------------------------------------
        # Compute lumi-weighted normalization factors
        # ------------------------------------------------------------------
        self.weights = self._compute_file_weights()
        self.structure_constants = None
        self.final_observables = {}

    # --------------------------------------------------------------
    def _compute_file_weights(self):
        """
        Compute per-file normalization weights using integrated luminosity (pb⁻¹)
        and the 'weightedEvents' histogram in each ROOT file.
        """

        INTEGRATED_LUMI = {
            "2016preVFP": 19500.0,
            "2016postVFP": 16810.0,
            "2017": 41480.0,
            "2018": 59830.0,
        }

        grouped = {r: [] for r in self.cross_sections}
        for path in self.file_paths:
            for region in grouped:
                if region in path:
                    grouped[region].append(path)

        weights = {}
        for region, files in grouped.items():
            if not files:
                continue

            # Collect total weighted events per era
            era_weighted = {}
            for f in files:
                try:
                    with uproot.open(f) as file:
                        if "weightedEvents" not in file:
                            print(f"[WARN] Missing 'weightedEvents' histogram in {f}")
                            continue
                        total_w = float(file["weightedEvents"].values()[0])
                        era_match = [e for e in INTEGRATED_LUMI if e in f]
                        if not era_match:
                            print(f"[WARN] Could not infer era for {f}")
                            continue
                        era = era_match[0]
                        era_weighted.setdefault(era, []).append(total_w)
                except Exception as e:
                    print(f"[WARN] Skipping file {f}: {e}")

            # Compute per-era normalization
            for era, wvals in era_weighted.items():
                total_w = np.sum(wvals)
                lumi = INTEGRATED_LUMI[era]
                xsec = self.cross_sections[region]
                if total_w <= 0:
                    print(f"[WARN] No valid weighted events for {region} ({era})")
                    continue
                scale = (lumi * xsec) / total_w
                for f in files:
                    if era in f:
                        weights[f] = scale
                print(f"[INFO] {region} ({era}): σ={xsec:.3f} pb, L={lumi:.0f} pb⁻¹, Σw={total_w:.3e} → scale={scale:.3e}")

        return weights


    # --------------------------------------------------------------
    def load_structure_constants(self):
        """Load structure constants (merged across all channels)."""
        valid_paths = [p for p in self.struct_paths if os.path.exists(p)]
        if not valid_paths:
            raise RuntimeError(" No structure constants found in provided directory.")

        arrays = [np.load(p) for p in tqdm(valid_paths, desc=f"Loading SC step{self.step}")]
        struct_array = np.concatenate(arrays, axis=0)
        self.structure_constants = torch.tensor(struct_array, dtype=torch.float32, device=self.device)
        print(f"→ Loaded {len(valid_paths)} SC files, combined shape={struct_array.shape}")

    # --------------------------------------------------------------
    def load_observables(self):
        """Read observable branches from ROOT files and apply selection."""
        collected = []
        total_before, total_after = 0, 0

        for fp in tqdm(self.file_paths, desc="Loading observables", unit="file"):
            try:
                with uproot.open(fp) as f:
                    tname = f"ttBar_treeVariables_step{self.step}"
                    if tname not in f:
                        print(f"[WARN] Skipping empty file: {fp}")
                        continue
                    t = f[tname]
                    extra = "trueLevelWeight" if self.step == 0 else "eventWeight"
                    keys = [k for k in t.keys()
                            if (self.step == 0 and k.startswith("gen_")) or (self.step == 8 and not k.startswith("gen_"))]
                    arrs = t.arrays(keys + [extra])
                    total_before += len(arrs[extra])

                    # Selection mask
                    if self.step == 0:
                        mask = (arrs["gen_l_pt"] > 0) & (arrs["gen_lbar_pt"] > 0)
                    else:
                        mask = (arrs["l_pt"] > 0) & (arrs["lbar_pt"] > 0)

                    #arrs = arrs[mask]
                    total_after += len(arrs[extra])
                    collected.append({k: arrs[k] for k in arrs.fields})
            except Exception as e:
                print(f"[WARN] Failed to load {fp}: {e}")
                continue

        if not collected:
            raise RuntimeError(" No valid observables loaded.")

        self.final_observables = {k: ak.concatenate([c[k] for c in collected]) for k in collected[0]}
        self.collected = collected

        print(f"[INFO] Total events before selection: {total_before}")
        print(f"[INFO] Total events after selection:  {total_after}")
        print(f"[INFO] Kept fraction: {100 * total_after / max(total_before, 1):.2f}%")

    # --------------------------------------------------------------
    def _base_weights(self):
        """Return lumi-weighted base weights per event."""
        ex_key = next(iter(self.collected[0]))
        weights_array = []
        for i, d in enumerate(self.collected):
            fp = self.file_paths[i]
            if fp not in self.weights:
                print(f"[WARN] No normalization weight for {fp}")
                continue
            weights_array.append(np.full(len(d[ex_key]), self.weights[fp]))
        if not weights_array:
            raise RuntimeError(" No valid base weights built.")
        return ak.concatenate(weights_array)

    # --------------------------------------------------------------
    def get_final_weights(self, wc_point):
        """Compute full EFT event weights, aligned with structure constants."""
        struct = self.structure_constants.cpu().numpy()
        n_struct = len(struct)
        n_obs = len(self.final_observables[next(iter(self.final_observables))])
        if n_struct != n_obs:
            n = min(n_struct, n_obs)
            print(f"[WARN] Structure constants ({n_struct}) and observables ({n_obs}) mismatch → truncating to {n}")
            struct = struct[:n]
            for key in list(self.final_observables.keys()):
                self.final_observables[key] = self.final_observables[key][:n]

        eft_w = Event_weight_prediction1.event_weights_lin_quad(struct, wc_point)[2] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        base_w = self._base_weights()

        if len(base_w) != len(eft_w):
            n = min(len(base_w), len(eft_w))
            print(f"[WARN] Adjusting array lengths: base={len(base_w)} vs eft={len(eft_w)} → truncating to {n}")
            base_w, eft_w = base_w[:n], eft_w[:n]

        return base_w * eft_w

    # --------------------------------------------------------------
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
        if self.step == 0:
            mask = (self.final_observables['gen_l_pt'] > 0) & (self.final_observables['gen_lbar_pt'] > 0)
        elif self.step == 8:
            mask = (self.final_observables['l_pt'] > 0) & (self.final_observables['lbar_pt'] > 0)
        else:
            raise ValueError("Step must be 0 (GEN) or 8 (RECO)")

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
