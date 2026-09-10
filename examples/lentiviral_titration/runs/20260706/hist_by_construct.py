#!/usr/bin/env python
"""Per-construct GFP ridgelines.

The `histograms` analysis in titration.toml groups by virus volume and so pools
all three constructs into one figure. Here we write one ridgeline per construct
(AA21/AA22/AA23), each showing its own dilution series plus the shared
untransduced control (0 uL) row, so the +/- split can be read per prep.

Reuses the library's load -> gate -> logicle pipeline and plot_histograms.
Run from this folder, after `gate-cells` has produced results/*_gate.json:
    uv run python hist_by_construct.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

from flowsmith import fluorescence as fl
from flowsmith.gating import apply_saved_gates, seed_populations
from flowsmith.io import load_samples
from flowsmith.style import rc

CHANNEL = "BL1-A"          # GFP reporter
CONTROL = "untransduced"
GROUP = "virus_uL"
PCT = 99.0                 # positive threshold = 99th percentile of the control

here = Path(__file__).parent
results = here / "results"
gate_paths = [results / "cells_gate.json", results / "singlets_gate.json"]

samples = load_samples(here / "samples.csv", here / "../../data/20260706", subsample=20000)
populations = seed_populations(samples)
singlets = apply_saved_gates(populations, gate_paths)

xform = fl.make_logicle()
# One shared threshold from the untransduced control, so every panel uses the
# same GFP+ cutoff.
threshold = fl.control_thresholds(singlets, [CHANNEL], xform, CONTROL, PCT).get(CHANNEL)

control_pop = [p for p in singlets if p["conditions"].get("construct") == "control"]
constructs = sorted({p["conditions"].get("construct") for p in singlets} - {"control"})

written = []
for c in constructs:
    pops = control_pop + [p for p in singlets if p["conditions"].get("construct") == c]
    n = fl.n_groups(pops, GROUP)
    height = max(4.0, 1.05 * n + 1.8)
    with mpl.rc_context(rc()):
        fig, ax = plt.subplots(figsize=(9, height))
        fl.plot_histograms(
            pops, CHANNEL, xform, ax=ax, group_col=GROUP,
            group_label="Virus (uL)", channel_label="GFP", threshold=threshold,
        )
        ax.set_title(c, fontweight="bold")
        path = results / f"hist_{c}_BL1-A.png"
        fig.savefig(path)
        plt.close(fig)
    written.append(path.name)

print("-> " + ", ".join(written))
