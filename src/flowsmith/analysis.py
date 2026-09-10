"""Composable, config-selectable fluorescence analyses.

Each analysis is a small function registered under a ``kind`` name. A config
file lists the analyses to run (with parameters); the runner builds an
:class:`AnalysisContext` from the gated populations and dispatches each one.

Add a new analysis by writing a function decorated with ``@register("name")``
that takes the context plus keyword parameters and writes its outputs to
``ctx.out_dir``. It then becomes available to every experiment's config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from flowsmith import fluorescence as fl
from flowsmith.style import INK, rc


@dataclass
class AnalysisContext:
    """Shared inputs handed to every analysis."""

    populations: list[dict]  # gated singlets, with .conditions + .events
    out_dir: Path
    xform: object  # logicle transform
    channels: list[str]  # default fluorescence channels of interest
    control_id: str | None  # default negative control sample id
    group_col: str | None  # default condition to colour/group by
    dose_col: str | None  # default numeric condition for dose-response
    per_sample: int = 20000
    positive_percentile: float = 99.0
    # Map a flow channel to the biology it reports (e.g. "BL1-A" -> "GFP"); used
    # for axis labels across all plots. Channels not in the map show as-is.
    channel_labels: dict = field(default_factory=dict)

    def label_for(self, channel: str) -> str:
        return self.channel_labels.get(channel, channel)


# -- registry -----------------------------------------------------------------

REGISTRY: dict[str, callable] = {}


def register(kind: str):
    def deco(fn):
        REGISTRY[kind] = fn
        return fn
    return deco


def run_analysis(kind: str, ctx: AnalysisContext, params: dict) -> str:
    if kind not in REGISTRY:
        raise ValueError(f"unknown analysis kind {kind!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[kind](ctx, **params)


def _safe(name: str) -> str:
    return name.replace("/", "_")


# -- column auto-detection ----------------------------------------------------

def _is_numeric(values) -> bool:
    try:
        [float(v) for v in values]
        return True
    except (TypeError, ValueError):
        return False


def detect_columns(populations, group_arg=None, dose_arg=None):
    """Auto-pick dose (numeric, varying) and group (categorical, varying) columns."""
    cond_keys = list(populations[0]["conditions"].keys())

    def uniques(key):
        return {p["conditions"][key] for p in populations}

    varying = [k for k in cond_keys if len(uniques(k)) > 1]

    dose = dose_arg
    if dose is None:
        numeric_varying = [k for k in varying if _is_numeric(uniques(k))]
        if "concentration" in numeric_varying:
            dose = "concentration"
        elif numeric_varying:
            dose = numeric_varying[0]

    group = group_arg
    if group is None:
        candidates = [k for k in varying if k != dose and not _is_numeric(uniques(k))]
        group = candidates[0] if candidates else None
    return group, dose


def detect_control(populations, control_arg=None, dose_col=None):
    if control_arg is not None:
        return control_arg
    if dose_col is not None:
        return min(populations, key=lambda p: float(p["conditions"][dose_col]))["sample_id"]
    return populations[0]["sample_id"]


def _replicate_n(stats_df, dose_col, group_col=None):
    """Typical replicate count per condition (None if it isn't uniform)."""
    keys = [dose_col] + ([group_col] if group_col else [])
    sizes = stats_df.groupby(keys).size()
    return int(sizes.iloc[0]) if sizes.nunique() == 1 else None


def dose_determines_group(populations, group_col, dose_col) -> bool:
    """True if each dose value maps to a single group value (group redundant)."""
    if group_col is None or dose_col is None:
        return False
    mapping = {}
    for p in populations:
        dose, grp = p["conditions"][dose_col], p["conditions"][group_col]
        if dose in mapping and mapping[dose] != grp:
            return False
        mapping[dose] = grp
    return True


# -- analyses -----------------------------------------------------------------

@register("mfi_pct")
def _mfi_pct(ctx, channels=None, control=None, positive_percentile=None, out="fluor_stats.csv"):
    """Per-sample MFI and % positive table."""
    channels = channels or ctx.channels
    control = control if control is not None else ctx.control_id
    pct = positive_percentile if positive_percentile is not None else ctx.positive_percentile
    stats, _ = fl.compute_stats(ctx.populations, channels, ctx.xform, control_id=control, positive_percentile=pct)
    path = ctx.out_dir / out
    stats.to_csv(path, index=False)
    return f"mfi_pct -> {path.name}"


@register("histograms")
def _histograms(ctx, channels=None, group=None, group_label=None, colors=None,
                replicate_mode="representative", control=None,
                positive_percentile=None, per_sample=None, order=None):
    """One logicle ridgeline PNG per channel, one ridge per condition value.

    ``group`` is the condition column to split on (auto-detected if omitted);
    ``group_label`` sets the y-axis title with units (e.g. ``"Dose (mM)"``).
    Colours auto-pick by data type (numeric -> sequential ramp, string ->
    categorical palette); ``colors`` overrides with an explicit list.
    ``replicate_mode`` is "representative" (one replicate per ridge) or "pool".
    ``order`` lists the group values top-to-bottom, overriding the default sort
    (e.g. to put a WT reference on the top ridge).
    """
    channels = channels or ctx.channels
    group = group if group is not None else ctx.group_col
    control = control if control is not None else ctx.control_id
    pct = positive_percentile if positive_percentile is not None else ctx.positive_percentile
    per_sample = per_sample or ctx.per_sample
    thresholds = (
        fl.control_thresholds(ctx.populations, channels, ctx.xform, control, pct) if control else {}
    )
    n = fl.n_groups(ctx.populations, group)
    height = max(4.0, 1.05 * n + 1.8)  # ridgeline grows with the number of rows
    written = []
    for ch in channels:
        with mpl.rc_context(rc()):
            fig, ax = plt.subplots(figsize=(9, height))
            fl.plot_histograms(
                ctx.populations, ch, ctx.xform, ax=ax, group_col=group,
                group_label=group_label, channel_label=ctx.label_for(ch),
                colors=colors, per_sample=per_sample,
                threshold=thresholds.get(ch), replicate_mode=replicate_mode,
                order=order,
            )
            path = ctx.out_dir / f"hist_{_safe(ch)}.png"
            fig.savefig(path)
            plt.close(fig)
        written.append(path.name)
    return "histograms -> " + ", ".join(written)


@register("density_2d")
def _density_2d(ctx, x=None, y=None, per_sample=None, qc=False, unit=None,
                xlabel=None, ylabel=None, cmap="viridis", row=None, col=None,
                row_order=None, col_order=None, out="density_2d.png"):
    """Faceted 2D logicle density.

    One panel per condition (representative replicate) by default; set
    ``qc = true`` for a QC view with one panel per sample. Give ``row`` and
    ``col`` (two condition columns) to lay the panels out as a grid — one
    condition down the rows, the other across the columns (``row_order`` /
    ``col_order`` set the sequence). ``unit`` is appended to each panel's
    condition value in the flat-tiling title. Axis labels default to the
    channel-label map (e.g. "BL1-A" -> "GFP"); ``xlabel`` / ``ylabel`` override.
    """
    x = x or ctx.channels[0]
    y = y or (ctx.channels[1] if len(ctx.channels) > 1 else ctx.channels[0])
    per_sample = per_sample or ctx.per_sample
    mode = "all" if qc else "representative"
    xlabel = xlabel or ctx.label_for(x)
    ylabel = ylabel or ctx.label_for(y)
    with mpl.rc_context(rc(scale=0.5)):  # small facet panels -> lighter weight
        fig = fl.plot_2d_density(ctx.populations, x, y, ctx.xform, per_sample=per_sample,
                                 mode=mode, unit=unit, xlabel=xlabel, ylabel=ylabel,
                                 cmap=cmap, row=row, col=col, row_order=row_order,
                                 col_order=col_order)
        path = ctx.out_dir / out
        fig.savefig(path)
        plt.close(fig)
    return f"density_2d -> {path.name}"


@register("dose_response")
def _dose_response(ctx, channels=None, dose=None, dose_label=None, group=None,
                   group_label=None, control=None, positive_percentile=None,
                   errorbar="sd", show_n=False, out="dose_response.png"):
    """MFI and % positive vs dose: per-replicate points + mean ± error curve.

    A grid of metrics (MFI, % positive) by channel. ``dose_label`` sets the
    shared x-axis title with units (e.g. ``"Dose (mM)"``); ``group_label``
    titles the per-group legend (e.g. ``"Cell line"`` for a ``cell_line``
    column); channel axes use the channel-label map (e.g. "BL1-A" -> "GFP").
    ``errorbar`` is ``"sd"`` (default), ``"sem"`` or ``"ci"`` (t-based 95 % CI).
    Set ``show_n = true`` to print a ``mean ± <error>, n = N`` note on the figure
    (off by default — usually stated in the figure caption instead).
    """
    channels = channels or ctx.channels
    dose = dose if dose is not None else ctx.dose_col
    if dose is None:
        return "dose_response -> skipped (no dose column)"
    group = group if group is not None else ctx.group_col
    control = control if control is not None else ctx.control_id
    pct = positive_percentile if positive_percentile is not None else ctx.positive_percentile
    stats, thresholds = fl.compute_stats(
        ctx.populations, channels, ctx.xform, control_id=control, positive_percentile=pct
    )
    dr_group = None if dose_determines_group(ctx.populations, group, dose) else group
    metrics = [("MFI", "MFI", False)] + (
        [("pct_pos", "% positive", True)] if thresholds else []
    )
    with mpl.rc_context(rc(scale=0.8)):
        fig, axes = plt.subplots(
            len(metrics), len(channels),
            figsize=(6.5 * len(channels), 5.5 * len(metrics)),
            squeeze=False, layout="constrained",
        )
        # Reserve a thin bottom strip for the figure-level n note (only when
        # shown) and a top strip for the shared group legend, so neither collides
        # with the panel labels.
        has_groups = dr_group is not None and stats[dr_group].nunique() > 1
        bottom = 0.035 if show_n else 0.0
        top = 0.915 if has_groups else 1.0
        fig.get_layout_engine().set(rect=(0, bottom, 1, top - bottom))
        for r, (prefix, metric_label, percent) in enumerate(metrics):
            for c, ch in enumerate(channels):
                fl.plot_dose_response(
                    axes[r][c], stats, dose, f"{prefix}_{ch}", group_col=dr_group,
                    group_label=group_label, dose_label=dose_label or dose,
                    percent=percent, show_n=False, show_legend=False, errorbar=errorbar,
                    y_label=f"{metric_label} ({ctx.label_for(ch)})",
                )
        # One shared group legend for the whole figure (the groups are the same in
        # every panel), instead of repeating it per panel.
        if has_groups:
            handles, lbls = axes[0][0].get_legend_handles_labels()
            leg = fig.legend(handles, lbls, title=group_label or dr_group, frameon=False,
                             loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=len(lbls))
            leg.get_title().set_fontweight("bold")
        # Optionally report the replicate count once for the whole figure (mean ±
        # SD of n biological replicates); off by default — usually in the caption.
        n = _replicate_n(stats, dose, dr_group) if show_n else None
        if n is not None:
            fig.text(0.99, 0.016, f"mean ± {fl.errorbar_label(errorbar)}, n = {n}",
                     ha="right", va="center", color=INK, fontweight="bold",
                     fontsize=plt.rcParams["xtick.labelsize"])
        path = ctx.out_dir / out
        fig.savefig(path)
        plt.close(fig)
    return f"dose_response -> {path.name}"


_VOLUME_TO_ML = {"ul": 1e-3, "uL": 1e-3, "µl": 1e-3, "µL": 1e-3, "ml": 1.0, "mL": 1.0}


@register("titer")
def _titer(ctx, cells_seeded=None, channel=None, volume=None, volume_unit="uL",
           control=None, positive_percentile=None, linear_min=5.0, linear_max=60.0,
           poisson=True, out="titer.csv"):
    """Functional viral titer (TU/mL) from a fluorescent-reporter dilution series.

    Titer = (target cells at transduction) x (transduced per cell) / (virus
    volume), so it needs ``cells_seeded`` (cells per well) — it cannot be derived
    from the cytometry alone. The reporter ``channel`` defaults to the first
    configured channel; ``volume`` is the per-well virus-volume column (defaults
    to the dose column) read in ``volume_unit`` (``"uL"`` default, or ``"mL"``).
    Only wells whose % positive falls in ``[linear_min, linear_max]`` (default
    5-60 %, the reliable single-integration regime — low enough to avoid the
    control's false-positive background, high enough to avoid saturation) feed the
    reported titer. By default
    each fraction is converted to MOI (``-ln(1 - f)``) before scaling, correcting
    the multiple-integration bias so you can safely widen ``linear_max`` (set
    ``poisson = false`` for the raw-fraction estimate). Writes a per-well CSV and
    a diagnostic plot, and returns the mean in-range titer.
    """
    if cells_seeded is None:
        raise ValueError("titer requires 'cells_seeded' (target cells per well at transduction).")
    channel = channel or ctx.channels[0]
    control = control if control is not None else ctx.control_id
    if control is None:
        raise ValueError("titer needs a 'control' (mock/untransduced well) to set the reporter+ threshold.")
    volume = volume if volume is not None else ctx.dose_col
    if volume is None:
        raise ValueError("titer needs a 'volume' column (per-well virus volume); set it or a dose column.")
    if volume_unit not in _VOLUME_TO_ML:
        raise ValueError(f"unknown volume_unit {volume_unit!r}; use one of {sorted(set(_VOLUME_TO_ML))}.")
    pct = positive_percentile if positive_percentile is not None else ctx.positive_percentile

    stats, _ = fl.compute_stats(ctx.populations, [channel], ctx.xform, control_id=control, positive_percentile=pct)
    titer_df = fl.compute_titer(
        stats, volume, f"pct_pos_{channel}", float(cells_seeded),
        volume_to_mL=_VOLUME_TO_ML[volume_unit], linear_min=linear_min,
        linear_max=linear_max, poisson=poisson,
    )
    csv_path = ctx.out_dir / out
    titer_df.rename(columns={"titer": "titer_TU_per_mL"}).to_csv(csv_path, index=False)

    in_range = titer_df[titer_df["in_range"]]
    mean_titer = float(in_range["titer"].mean()) if not in_range.empty else float("nan")
    with mpl.rc_context(rc()):
        fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
        fl.plot_titer(ax, titer_df, volume, mean_titer=mean_titer,
                      volume_label=f"Virus volume ({volume_unit})")
        plot_path = ctx.out_dir / f"{csv_path.stem}.png"
        fig.savefig(plot_path)
        plt.close(fig)

    if in_range.empty:
        return (f"titer -> {csv_path.name}, {plot_path.name} "
                f"(no wells in {linear_min:g}-{linear_max:g}% range; widen linear_min/linear_max)")
    return (f"titer -> {csv_path.name}, {plot_path.name} "
            f"(mean {mean_titer:.2e} TU/mL over {len(in_range)} wells)")


def _parse_sig(sig):
    """Config significance pairs -> ``((group, subgroup), (group, subgroup))`` list.

    Each comparison is a 2-list of endpoints; an endpoint is a bare group value
    (no subgroup) or a ``[group, subgroup]`` pair.
    """
    def endpoint(e):
        if isinstance(e, (list, tuple)):
            return (e[0], e[1] if len(e) > 1 else None)
        return (e, None)
    return [(endpoint(a), endpoint(b)) for a, b in sig]


@register("categorical")
def _categorical(ctx, channels=None, group=None, group_label=None, subgroup=None,
                 subgroup_label=None, control=None, positive_percentile=None, sig=None,
                 order=None, subgroup_order=None, errorbar="sd", out="categorical.png"):
    """Categorical comparison for unordered groups (e.g. CAR_A vs CAR_B).

    A grid of metrics (MFI, % positive) by channel; the ``group`` condition is
    the categorical x-axis (auto-detected if omitted), with jittered replicate
    points + mean ± error (``errorbar`` = ``"sd"`` default / ``"sem"`` / ``"ci"``
    t-based 95 % CI). An optional ``subgroup`` condition dodges each group into
    shaped/coloured series with a shared legend. ``sig`` lists pairs to mark with
    a two-sample t-test bracket (``*/**/***/ns``).
    """
    channels = channels or ctx.channels
    group = group if group is not None else ctx.group_col
    if group is None:
        return "categorical -> skipped (no categorical group column)"
    control = control if control is not None else ctx.control_id
    pct = positive_percentile if positive_percentile is not None else ctx.positive_percentile
    stats, thresholds = fl.compute_stats(
        ctx.populations, channels, ctx.xform, control_id=control, positive_percentile=pct
    )
    comparisons = _parse_sig(sig) if sig else None
    has_sub = subgroup is not None and subgroup in stats and stats[subgroup].nunique() > 1
    metrics = [("MFI", "MFI", False)] + (
        [("pct_pos", "% positive", True)] if thresholds else []
    )
    with mpl.rc_context(rc(scale=0.8)):
        fig, axes = plt.subplots(
            len(metrics), len(channels),
            figsize=(6.5 * len(channels), 5.5 * len(metrics)),
            squeeze=False, layout="constrained",
        )
        if has_sub:  # reserve a top strip for the shared subgroup legend
            fig.get_layout_engine().set(rect=(0, 0, 1, 0.915))
        for r, (prefix, metric_label, percent) in enumerate(metrics):
            for c, ch in enumerate(channels):
                fl.plot_categorical(
                    axes[r][c], stats, group, f"{prefix}_{ch}",
                    subgroup_col=subgroup if has_sub else None,
                    group_label=group_label or group, order=order,
                    subgroup_order=subgroup_order, comparisons=comparisons,
                    percent=percent, errorbar=errorbar,
                    y_label=f"{metric_label} ({ctx.label_for(ch)})",
                )
        if has_sub:
            handles, lbls = axes[0][0].get_legend_handles_labels()
            leg = fig.legend(handles, lbls, title=subgroup_label or subgroup, frameon=False,
                             loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=len(lbls))
            leg.get_title().set_fontweight("bold")
        path = ctx.out_dir / out
        fig.savefig(path)
        plt.close(fig)
    return f"categorical -> {path.name}"


@register("quadrant")
def _quadrant(ctx, x, y, control=None, positive_percentile=None, labels=None,
              dose=None, dose_label=None, unit=None, xlabel=None, ylabel=None,
              legend_title="Population", cmap="viridis", show_n=False, errorbar="sd",
              row=None, col=None, row_order=None, col_order=None, out="quadrant.csv"):
    """Two-threshold quadrant analysis: % in each of four populations.

    Writes a per-sample CSV, a faceted density plot (representative replicate
    per condition) with crosshairs and per-quadrant labels, and (if a dose
    column is available) a per-quadrant dose-response with mean ± error
    (``errorbar`` = ``sd``/``sem``/``ci``) across replicates. ``unit`` labels the
    density panels; give ``row`` and ``col`` to lay them out as a condition grid
    (``row_order`` / ``col_order`` set the
    sequence); ``dose_label`` titles the dose axis; ``xlabel``/``ylabel`` default
    to the channel-label map.
    """
    control = control if control is not None else ctx.control_id
    pct = positive_percentile if positive_percentile is not None else ctx.positive_percentile
    thr = fl.control_thresholds(ctx.populations, [x, y], ctx.xform, control, pct)
    quad = fl.quadrant_stats(ctx.populations, x, y, ctx.xform, thr[x], thr[y], labels=labels)
    csv_path = ctx.out_dir / out
    quad.to_csv(csv_path, index=False)
    written = [csv_path.name]

    with mpl.rc_context(rc(scale=0.5)):  # small facet panels -> lighter weight
        fig = fl.plot_quadrants(
            ctx.populations, x, y, ctx.xform, thr[x], thr[y], quad, labels=labels,
            per_sample=ctx.per_sample, unit=unit, cmap=cmap, row=row, col=col,
            row_order=row_order, col_order=col_order,
            xlabel=xlabel or ctx.label_for(x), ylabel=ylabel or ctx.label_for(y),
        )
        plot_path = ctx.out_dir / f"{csv_path.stem}_density.png"
        fig.savefig(plot_path)
        plt.close(fig)
    written.append(plot_path.name)

    dose = dose if dose is not None else ctx.dose_col
    if dose is not None:
        with mpl.rc_context(rc(scale=0.9)):
            fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
            if show_n:
                fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.93))  # bottom strip for n
            fl.plot_quadrant_dose_response(
                ax, quad, dose, labels=labels, dose_label=dose_label or dose,
                legend_title=legend_title, show_n=False, errorbar=errorbar,
            )
            n = _replicate_n(quad, dose) if show_n else None
            if n is not None:
                fig.text(0.99, 0.016, f"mean ± {fl.errorbar_label(errorbar)}, n = {n}",
                         ha="right", va="center", color=INK, fontweight="bold",
                         fontsize=plt.rcParams["xtick.labelsize"])
            dr_path = ctx.out_dir / f"{csv_path.stem}_dose_response.png"
            fig.savefig(dr_path)
            plt.close(fig)
        written.append(dr_path.name)
    return "quadrant -> " + ", ".join(written)
