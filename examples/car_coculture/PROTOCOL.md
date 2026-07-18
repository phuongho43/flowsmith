# CAR-Jurkat activation co-culture — protocol & analysis guide

A step-by-step guide to functionally screening **CAR constructs** by the
activation they drive when their Jurkat effectors meet antigen-positive target
cells, and to running the flowsmith pipeline on the resulting FCS files. Aimed at
a bench biologist; no prior flowsmith experience assumed.

**Scope: a fast functional screen.** Jurkat is an immortalised T-cell line that
**upregulates CD69** (an early activation marker) through the same TCR/CAR
signalling pathway as primary T cells, but it does **not kill** targets. So this
assay reports **"does the CAR signal on antigen?"** and ranks constructs by
signalling strength — it is **not** a cytotoxicity/killing assay, and Jurkat
activation does not always predict primary-T-cell potency. It is a quick, cheap
triage of a construct panel before committing to primary cells.

The idea: co-culture each **CAR-Jurkat line** (which carries a **GFP** marker) with
a **titration of antigen-positive target cells** (here **OVCAR3**), and a few hours
later measure **CD69** on the CAR-Jurkats by flow. A functional CAR turns CD69 up
when targets are present; a non-functional one stays at baseline.

---

## Part A — Wet lab

**You will need:** the CAR-Jurkat lines (here **AA21 / AA22 / AA23**, each
CAR-2A-GFP so transduced cells are GFP+), **untransduced (WT) Jurkats** as the
negative control, the antigen-positive target line (**OVCAR3**), a round-bottom
96-well plate, an anti-**CD69** antibody (here **APC**-conjugated, read in the
"RL1"/APC channel), and a cytometer with a 488 nm (GFP → "BL1"/FITC) and a 638 nm
(APC → "RL1") laser line.

1. **Plate a fixed number of effectors.** Add a constant number of Jurkats per
   well (e.g. **1 × 10⁵**) for every construct and every condition — the effector
   count is the "1" in the E:T ratio and must not vary.

2. **Titrate the targets (the E:T series).** Add OVCAR3 targets to give
   **effector : target (E:T) ratios of 1:0, 1:1, 1:2, 1:4** — i.e. 0, 1, 2 and 4
   targets per Jurkat. The **1:0 (Jurkat-only, no target)** well is essential: it
   is each line's own no-antigen baseline and reveals any **tonic** (antigen-
   independent) CAR signalling. Run the **full E:T series for every construct and
   for WT** (4 constructs × 4 ratios = 16 wells).

3. **Co-incubate 4–24 h.** CD69 is an early marker — 4–6 h already shows a clear
   shift; overnight is fine. Longer co-cultures let targets proliferate and dilute
   the ratio, so keep the timing consistent across wells.

4. **Stain and read.** Surface-stain with anti-CD69, wash, and acquire on the
   cytometer. Record **≥20,000 events/well**. GFP (CAR marker) and CD69 do not
   need compensation if their spectra are well separated; add single-stain
   controls if you see spillover.

> **Why WT?** Untransduced Jurkats have no CAR and no GFP. Their **1:0 well** is
> the true CD69-negative baseline that sets the `CD69+` threshold, and running WT
> across the E:T series confirms that any activation you see is **CAR-dependent**,
> not a generic response to co-culture.

---

## Part B — Files and conditions

Put the FCS files in `data/` and describe them in `samples.csv`. The files here
are nested one folder per E:T ratio (`1-to-0/`, `1-to-1/`, …), so the `filename`
column carries that sub-path:

```csv
filename,sample,construct,jurkat,et_ratio,target_n
1-to-1/..._AA21-1to1.fcs,AA21_1-1,AA21,CAR,1:1,1
1-to-1/..._WT-1to1.fcs,WT_1-1,WT,untransduced,1:1,1
```

- **`construct`** — the CAR line (`AA21`/`AA22`/`AA23`) or `WT`. The plot series.
- **`jurkat`** — `CAR` (GFP+, gate on GFP) vs `untransduced` (WT, no GFP gate).
- **`et_ratio`** / **`target_n`** — the E:T label (`1:2`) and its numeric targets-
  per-Jurkat (`2`), used as the x-axis.

---

## Part C — Analysis

The readout is CD69 on the **CAR-Jurkat effectors**. Jurkats and OVCAR3 overlap in
scatter, so the effectors are isolated by their **GFP** marker rather than by
FSC/SSC. This needs a GFP+ sub-gate on top of the standard cells+singlets gates,
so the analysis is a short script (`car_activation.py`) rather than an `analyze`
TOML.

1. **Draw the shared scatter gates once.** Auto-gating is hands-off; always check
   the overlays. The auto cells-ellipse can sit low if the main cluster rides high
   on SSC — nudge it up if so.

   ```bash
   uv run gate-cells --sheet samples.csv --data data --out results
   ```

2. **Run the activation analysis.**

   ```bash
   uv run python car_activation.py
   ```

   It replays the gates, then for each sample:
   - **CAR lines:** gate **GFP+** (BL1-A above the 99th percentile of WT 1:0) to
     keep only CAR-Jurkats, and read CD69 on them.
   - **WT:** no GFP+ effectors to gate, so read CD69 on the whole singlet
     population (it is the negative control, not an effector readout).

   Both the **GFP+** and **CD69+** thresholds are the 99th percentile of the
   **WT 1:0** well (untransduced resting Jurkats — the true negative). Using WT for
   the CD69+ line means the no-target (1:0) CAR-Jurkats show their **tonic leakage**
   rather than being defined as 0%.

### Outputs (in `results/`)

| File | Contents |
|------|----------|
| `car_cd69_stats.csv` | Per-sample CD69 MFI + % positive, the population read (`GFP+` vs `singlets`), and n |
| `car_dose_response.png` | CD69 **MFI** and **% positive** vs E:T, one line per CAR construct |
| `car_cd69_ridgeline.png` | CD69 distributions per construct × E:T (WT first as the negative), with the shared `CD69+` line and per-ridge % |

### Reading the result

A **functional CAR** (AA21 here) shows CD69 **rising sharply once targets are
present** (1:0 → 1:1) and plateauing — the ridgeline distribution marches right.
**Non-functional CARs** (AA22, AA23) stay flat at their 1:0 baseline regardless of
target, and **WT** sits below all of them near 0%. Confirm all CAR lines are
comparably transduced (`n` of GFP+ cells) so a flat response reflects the CAR, not
missing effectors.

**Caveats** (see also the repo's `ASSUMPTIONS.md`): one well per condition (no
replicates, no error bars); Jurkat CD69 is a signalling proxy, not killing; and
the WT co-culture wells (1:1–1:4) are ungated mixtures of WT Jurkats + OVCAR3, so
only their 1:0 well is a clean Jurkat baseline.
