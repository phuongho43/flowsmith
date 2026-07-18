# Example configs

Ready-made, annotated templates for common flow-cytometry experiments. Each
subfolder is a self-contained template: an analysis (a `*.toml` config for the
config-driven `analyze` runner, or a short `*.py` script when the experiment
needs a sub-gate the runner doesn't do) plus an example `samples.csv` showing the
condition columns it expects. Copy a folder, drop your FCS files into its `data/`
directory, adjust the channels/conditions, and run.

Every template assumes the standard two-step flowsmith workflow — draw the
shared FSC/SSC gates once, then run the config-driven analysis:

```bash
cd examples/lentiviral_titration
uv run gate-cells --sheet samples.csv --data data --out results   # once
uv run analyze titration.toml                                     # the analyses
```

Channels in the templates use Attune-style names (`BL1-A` = GFP). Rename them to
your instrument's channels and update the `[channel_labels]` table to match.

## Available templates

| Folder | Experiment | Key readout |
|--------|------------|-------------|
| [`lentiviral_titration/`](lentiviral_titration/) | Serial-dilution titration of a fluorescent-reporter lentivirus | functional titer (TU/mL) + % reporter+ vs virus volume |
| [`car_coculture/`](car_coculture/) | CAR-Jurkat activation vs an antigen-target (OVCAR3) E:T titration | CD69 MFI + % positive on GFP+ CAR-Jurkats vs E:T ratio, per construct |

Both templates include a full wet-lab + analysis walkthrough (`PROTOCOL.md` in
each folder). The `car_coculture` analysis is a short script
([`car_activation.py`](car_coculture/car_activation.py)) rather than a TOML,
because it gates GFP+ CAR-Jurkats before reading the CD69 activation marker.

More templates (dose-response titrations, induction time-courses, …) to follow.
