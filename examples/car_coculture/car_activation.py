#!/usr/bin/env python
"""CAR-Jurkat activation (CD69) vs E:T ratio, per CAR construct.

Assay: CAR-Jurkats (BL1-A / CAR-GFP+) are co-cultured with OVCAR3 targets at
E:T = 1:0 (Jurkat only), 1:1, 1:2, 1:4. Engaging antigen upregulates CD69
(RL1-A). Jurkats and OVCAR3 overlap in scatter, so the CAR-Jurkat effectors are
isolated by a GFP+ gate (untransduced WT Jurkats and OVCAR3 are GFP-negative);
CD69 is then read on those GFP+ cells.

Gating hierarchy (beyond the shared cells+singlets gates from gate-cells):
  cells -> singlets -> GFP+ (CAR-Jurkats) -> CD69 readout

Thresholds (both 99th percentile of the WT 1:0 well = untransduced resting
Jurkats, the true negative):
  * GFP+   = WT_1-0's BL1-A -> isolates CAR-Jurkats (WT/OVCAR3 are GFP-negative).
  * CD69+  = WT_1-0's RL1-A -> a shared CD69+ line for every panel, so the
             no-target (1:0) CAR-Jurkats reveal their tonic activation leakage.

WT is untransduced (no GFP+ effectors to gate), so its CD69 is read on the whole
singlet population; it anchors the ridgeline as the negative control and is not
in the CAR-only dose-response.

This is the reusable template: drop your FCS into ./data/ (in E:T subfolders that
match samples.csv), then:
    uv run gate-cells --sheet samples.csv --data data --out results
    uv run python car_activation.py
See runs/20260710/ for the worked example this ships with.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

from flowsmith import fluorescence as fl
from flowsmith.gating import apply_saved_gates, seed_populations
from flowsmith.io import load_samples
from flowsmith.style import rc

GFP = "BL1-A"      # CAR-GFP (transduction marker)
CD69 = "RL1-A"     # activation marker
PCT = 99.0

here = Path(__file__).parent
results = here / "results"
gate_paths = [results / "cells_gate.json", results / "singlets_gate.json"]

samples = load_samples(here / "samples.csv", here / "data", subsample=20000)
singlets = apply_saved_gates(seed_populations(samples), gate_paths)
xform = fl.make_logicle()


def transformed(pop, ch):
    return xform.apply(pop["events"][ch].to_numpy(dtype=float))


# --- GFP+ gate: isolate CAR-Jurkats -----------------------------------------
gfp_thresh = fl.control_thresholds(singlets, [GFP], xform, "WT_1-0", PCT)[GFP]
gfp_pos = []
for p in singlets:
    mask = transformed(p, GFP) > gfp_thresh
    gfp_pos.append({**p, "events": p["events"][mask]})

# --- CD69+ threshold: untransduced resting WT Jurkats (WT_1-0) ----------------
# The true CD69-negative baseline is untransduced Jurkats, not the resting CAR-
# Jurkats: setting the threshold from the CAR 1:0 cells would define their own
# tonic/background signaling as "negative". Using WT lets the 1:0 (no-target)
# CAR groups reveal their activation leakage.
cd69_thresh = fl.control_thresholds(singlets, [CD69], xform, "WT_1-0", PCT)[CD69]

# --- Population each sample's CD69 is read on --------------------------------
# CAR samples: GFP+ CAR-Jurkat effectors. WT: the whole singlet population (no
# GFP+ to gate on — WT is untransduced), i.e. ungated Jurkats (+ any OVCAR3 in
# the co-culture wells). singlets and gfp_pos share order.
read_pops = [
    p_gfp if p_s["conditions"]["jurkat"] == "CAR" else p_s
    for p_s, p_gfp in zip(singlets, gfp_pos)
]

# --- Per-sample CD69 stats ---------------------------------------------------
rows = []
for p in read_pops:
    c = p["conditions"]
    raw = p["events"][CD69].to_numpy(dtype=float)
    xf = transformed(p, CD69)
    rows.append({
        "sample_id": p["sample_id"], "construct": c["construct"],
        "jurkat": c["jurkat"], "et_ratio": c["et_ratio"], "target_n": c["target_n"],
        "gated": "GFP+" if c["jurkat"] == "CAR" else "singlets",
        "n_read": len(raw),
        "MFI_CD69": round(float(np.median(raw)), 1) if len(raw) else np.nan,
        "pct_CD69": round(100 * float(np.mean(xf > cd69_thresh)), 2) if len(xf) else np.nan,
    })
stats = pd.DataFrame(rows).sort_values(["construct", "target_n"])
stats.to_csv(results / "car_cd69_stats.csv", index=False)
print(f"GFP+ threshold (raw≈{xform.inverse(np.array([gfp_thresh]))[0]:.0f}), "
      f"CD69+ threshold (raw≈{xform.inverse(np.array([cd69_thresh]))[0]:.0f})")

# CAR constructs only for the dose-response (WT has no CAR-Jurkat population).
car = stats[stats["jurkat"] == "CAR"].copy()

# --- Dose-response: CD69 MFI + % positive vs E:T, one series per construct -----
with mpl.rc_context(rc()):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), layout="constrained")
    fl.plot_dose_response(axes[0], car, "target_n", "MFI_CD69", group_col="construct",
                          group_label="CAR construct", y_label="CD69 MFI (RL1-A)", logx=False)
    fl.plot_dose_response(axes[1], car, "target_n", "pct_CD69", group_col="construct",
                          group_label="CAR construct", y_label="CD69+ (%)", percent=True,
                          logx=False, show_legend=False)
    for ax in axes:
        ax.set_xticks([0, 1, 2, 4])
        ax.set_xticklabels(["1:0", "1:1", "1:2", "1:4"])  # label ticks as E:T ratios
        ax.set_xlabel("")
    fig.supxlabel("OVCAR3 targets per Jurkat (E:T)", fontweight="bold")
    fig.savefig(results / "car_dose_response.png")
    plt.close(fig)

# --- CD69 ridgelines, one panel per construct --------------------------------
# CAR constructs are the GFP+ CAR-Jurkats; WT is the ungated singlets (its 1:0
# well is the untransduced negative that sets the shared threshold line).
constructs = ["WT", "AA21", "AA22", "AA23"]
titles = {"AA21": "AA21 CAR-Jurkats", "AA22": "AA22 CAR-Jurkats",
          "AA23": "AA23 CAR-Jurkats", "WT": "WT Jurkats (untransduced, ungated)"}
with mpl.rc_context(rc()):
    fig, axes = plt.subplots(len(constructs), 1, figsize=(9.5, 3.7 * len(constructs)),
                             sharex=True, layout="constrained")
    for ax, con in zip(axes, constructs):
        pops = [p for p in read_pops if p["conditions"]["construct"] == con]
        fl.plot_histograms(pops, CD69, xform, ax=ax, group_col="et_ratio",
                           group_label="E:T ratio", channel_label="CD69", threshold=cd69_thresh)
        ax.set_title(titles[con], fontweight="bold")
        ax.set_xlabel("")
        # Single-line E:T + n labels (4 ridges are too tight for a two-line label).
        order = sorted({p["conditions"]["et_ratio"] for p in pops})
        n_of = {g: sum(len(p["events"]) for p in pops if p["conditions"]["et_ratio"] == g)
                for g in order}
        ax.set_yticklabels([f"{g}  (n={n_of[g]:,})" for g in order])
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(plt.rcParams["ytick.labelsize"] * 0.78)
    axes[-1].set_xlabel("CD69")
    fig.savefig(results / "car_cd69_ridgeline.png")
    plt.close(fig)

print("-> car_cd69_stats.csv, car_dose_response.png, car_cd69_ridgeline.png")
