#!/usr/bin/env python
"""Per-construct functional titer for a multi-prep titration.

The built-in `titer` analysis averages over every in-range well, which is only
right for a single virus prep. Here AA21/AA22/AA23 are three independent preps
sharing one untransduced control, so each gets its own titer. This reuses the
library's compute_titer / plot_titer on results/fluor_stats.csv (produced by
`analyze titration.toml`), one construct at a time.

Run from this folder, after `analyze titration.toml`:
    uv run python titer_by_construct.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from flowsmith import fluorescence as fl
from flowsmith.style import rc

CELLS_SEEDED = 100_000     # 0.1M HEK293T per well at transduction
CHANNEL = "BL1-A"          # GFP reporter
VOLUME_COL = "virus_uL"
LINEAR_MIN, LINEAR_MAX = 5.0, 60.0

here = Path(__file__).parent
stats = pd.read_csv(here / "results" / "fluor_stats.csv")
constructs = [c for c in stats["construct"].unique() if c != "control"]

rows, panels = [], []
for c in sorted(constructs):
    sub = stats[stats["construct"] == c]
    titer_df = fl.compute_titer(
        sub, VOLUME_COL, f"pct_pos_{CHANNEL}", float(CELLS_SEEDED),
        linear_min=LINEAR_MIN, linear_max=LINEAR_MAX, poisson=True,
    )
    in_range = titer_df[titer_df["in_range"]]
    mean_titer = float(in_range["titer"].mean()) if not in_range.empty else float("nan")
    rows.append(titer_df)
    panels.append((c, titer_df, mean_titer, len(in_range)))

combined = pd.concat(rows, ignore_index=True).rename(columns={"titer": "titer_TU_per_mL"})
combined.to_csv(here / "results" / "titer_by_construct.csv", index=False)

import matplotlib as mpl
for c, titer_df, mean_titer, n in panels:
    with mpl.rc_context(rc()):
        fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.90))  # top strip for the key
        fl.plot_titer(ax, titer_df, VOLUME_COL, mean_titer=mean_titer,
                      volume_label="Virus volume (uL)")
        if ax.get_legend() is not None:  # lift the key clear of the title
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                       ncol=len(labels), frameon=False)
        ax.set_title(c, fontweight="bold")
        fig.savefig(here / "results" / f"titer_{c}.png")
        plt.close(fig)

print(f"{'construct':<10} {'mean TU/mL':>12} {'n in-range':>10}")
for c, _, mean_titer, n in panels:
    print(f"{c:<10} {mean_titer:>12.2e} {n:>10}")
print("-> results/titer_by_construct.csv, " +
      ", ".join(f"results/titer_{c}.png" for c, *_ in panels))
