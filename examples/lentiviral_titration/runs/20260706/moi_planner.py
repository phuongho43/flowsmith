#!/usr/bin/env python
"""MOI planning helper: how much of each prep to add per N cells at a target MOI.

Reads the per-construct titers (results/titer_by_construct.csv, mean over the
in-range wells) and, for a grid of target cell numbers x MOIs, computes the
virus volume needed:

    TU_needed = N_cells * MOI
    uL_needed = TU_needed / titer_TU_per_mL * 1000

Cells that need more than AVAILABLE_UL (your 100 uL prep) are flagged infeasible
(you'd have to make more virus or accept a lower MOI). Writes a tidy CSV, prints
per-construct tables, and renders a colour-coded grid.

Run from this folder, after titer_by_construct.py:
    uv run python moi_planner.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flowsmith.style import INK, SEQUENTIAL_CMAP, rc

AVAILABLE_UL = 100.0        # prep resuspended in 100 uL PBS
CELL_NUMBERS = [1e5, 2.5e5, 5e5, 1e6, 2e6]
MOIS = [0.3, 1, 3, 5, 10]

here = Path(__file__).parent
results = here / "results"

titer_df = pd.read_csv(results / "titer_by_construct.csv")
in_range = titer_df[titer_df["in_range"]]
titers = (in_range.groupby("construct")["titer_TU_per_mL"].mean()
          .sort_index().to_dict())


def uL_needed(n_cells, moi, titer):
    return n_cells * moi / titer * 1000.0  # TU / (TU/mL) -> mL -> uL


# --- tidy CSV -----------------------------------------------------------------
rows = []
for c, titer in titers.items():
    for n in CELL_NUMBERS:
        for moi in MOIS:
            v = uL_needed(n, moi, titer)
            rows.append({
                "construct": c, "titer_TU_per_mL": titer,
                "cells": int(n), "MOI": moi,
                "TU_needed": n * moi, "uL_needed": round(v, 2),
                "feasible_100uL": v <= AVAILABLE_UL,
            })
plan = pd.DataFrame(rows)
plan.to_csv(results / "moi_plan.csv", index=False)


def fmt(v):
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


# --- console tables -----------------------------------------------------------
for c, titer in titers.items():
    print(f"\n{c}  (titer {titer:.2e} TU/mL, {AVAILABLE_UL:g} uL available)")
    print("  uL to add   |  " + "  ".join(f"{int(n):>8,}" for n in CELL_NUMBERS) + "   cells")
    print("  " + "-" * (13 + 10 * len(CELL_NUMBERS)))
    for moi in MOIS:
        cells = []
        for n in CELL_NUMBERS:
            v = uL_needed(n, moi, titer)
            cells.append(f"({fmt(v)})" if v > AVAILABLE_UL else f"{fmt(v)}")
        print(f"  MOI {moi:>4}    |  " + "  ".join(f"{x:>8}" for x in cells))
print(f"\n(parentheses) = needs more than your {AVAILABLE_UL:g} uL prep")

# --- figure: one grid per construct ------------------------------------------
# Fill encodes uL magnitude (single-hue sequential ramp, log-scaled, shared
# across panels); a diagonal hatch + parenthesised value marks cells that exceed
# the available prep (status by texture + label, not colour alone).
lo, hi = np.log10(0.3), np.log10(700)
with mpl.rc_context(rc(scale=0.72)):
    fig, axes = plt.subplots(1, len(titers), figsize=(6.2 * len(titers), 6.4),
                             squeeze=False, sharey=True, layout="constrained")
    for ax, (c, titer) in zip(axes[0], titers.items()):
        grid = np.array([[uL_needed(n, moi, titer) for n in CELL_NUMBERS]
                         for moi in MOIS])
        norm = np.clip((np.log10(grid) - lo) / (hi - lo), 0, 1)
        ax.imshow(norm, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1, aspect="auto")
        for i, moi in enumerate(MOIS):
            for j, n in enumerate(CELL_NUMBERS):
                v = grid[i, j]
                over = v > AVAILABLE_UL
                # white text on the dark (high-uL) end, ink on the light end
                txt = "white" if norm[i, j] > 0.62 else INK
                ax.text(j, i, f"({fmt(v)})" if over else fmt(v), ha="center",
                        va="center", color=txt, fontweight="bold",
                        fontsize=plt.rcParams["font.size"] * (0.85 if over else 1.0))
                if over:
                    ax.add_patch(mpl.patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                        edgecolor="white", linewidth=0, alpha=0.55))
        ax.set_xticks(range(len(CELL_NUMBERS)))
        ax.set_xticklabels([f"{int(n/1000)}k" if n < 1e6 else f"{n/1e6:g}M"
                            for n in CELL_NUMBERS])
        ax.set_yticks(range(len(MOIS)))
        ax.set_yticklabels([f"{m:g}" for m in MOIS])
        ax.set_xlabel("Target cells")
        ax.set_title(f"{c}\n{titer:.1e} TU/mL", fontsize=plt.rcParams["axes.titlesize"] * 0.8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0][0].set_ylabel("Target MOI")
    fig.suptitle(f"Virus volume to add (uL)   —   hatched = exceeds {AVAILABLE_UL:g} uL prep",
                 fontweight="bold", fontsize=plt.rcParams["axes.titlesize"] * 0.8)
    fig.savefig(results / "moi_plan.png")
    plt.close(fig)

print("-> results/moi_plan.csv, results/moi_plan.png")
