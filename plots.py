"""Figures and tables for the SEAFL vs Libra comparison, built from saved result files only.

Sits at the end of the pipeline (requirements section 7): ``flsim.run`` writes
``results/<exp>/<run_name>/<seed>/{config.json,evals.csv,events.jsonl,summary.json}`` (formats in
``flsim/metrics.py``) and this script turns them into PNG + PDF figures and CSV + Markdown tables
under ``results/<exp>/figures/``. It never imports ``flsim`` or touches a model, so plots can be
regenerated on a laptop long after the GPU box is gone. The layout without a seed level
(``results/<exp>/<run_name>/evals.csv``, used by the smoke runs) is accepted and treated as one
seed.

Usage:
    python plots.py results/<exp> [--target 0.8]
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: results are plotted on servers and in CI without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# IEEE single column is ~3.5 in wide; every figure uses it so fonts stay legible in the report.
FIG_W = 3.5
TARGET_FRACTION = 0.9  # D8: target = 90% of the best accuracy any run reaches
LAST_K = 5  # evals used for the accuracy-fluctuation std
GRID_POINTS = 200

# One fixed colour per algorithm family (colourblind-safe: grey / blues / oranges).
FAMILY_COLOURS = {
    "fedavg": ["#7f7f7f"],
    "seafl": ["#08519c", "#3182bd", "#6baed6", "#9ecae1"],
    "libra": ["#a63603", "#e6550d", "#fd8d3c", "#fdae6b"],
    "other": ["#117733", "#882255", "#44aa99", "#aa4499"],
}


@dataclass
class Run:
    """All saved files of one (run_name, seed) pair.

    Attributes:
        seed: Seed directory name (``"0"`` for the layout without a seed level).
        evals: Column name -> float array from ``evals.csv`` (empty if the file is missing).
        events: Parsed lines of ``events.jsonl`` (empty if missing).
        summary: Parsed ``summary.json`` (empty if missing).
        config: Parsed ``config.json`` (empty if missing).
    """

    seed: str
    evals: dict[str, np.ndarray] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_plots.py)
# ----------------------------------------------------------------------------------------------


def time_to_target(t: np.ndarray, acc: np.ndarray, target: float) -> float | None:
    """First time at which the accuracy curve reaches the target.

    Args:
        t: Eval times in seconds, ascending.
        acc: Test accuracy at each time.
        target: Accuracy threshold in [0, 1].

    Returns:
        The first ``t`` with ``acc >= target``, or None if the run never reaches it.
    """
    hit = np.flatnonzero(np.asarray(acc) >= target)
    return float(t[hit[0]]) if hit.size else None


def interp_to_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate a curve onto a common grid so seeds can be averaged.

    Args:
        x: Curve abscissa (seconds or message count), non-decreasing.
        y: Curve values.
        grid: Target abscissa values.

    Returns:
        Values on ``grid``; NaN where the grid lies beyond the curve's last point, so a run that
        stopped early does not get extrapolated (and does not bias the mean).
    """
    return np.interp(grid, x, y, left=np.nan, right=np.nan)


def mean_std(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population std over seeds, ignoring NaN padding.

    Args:
        curves: One equally long array per seed.

    Returns:
        ``(mean, std)``; std is all zeros for a single seed so no band is drawn.
    """
    stack = np.vstack(curves)
    with warnings.catch_warnings():
        # All-NaN columns (grid past every seed's end) are expected and simply stay NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)


def fluctuation(acc: np.ndarray, last: int = LAST_K) -> float:
    """Accuracy fluctuation: std of the last ``last`` evals (requirements section 5).

    Args:
        acc: Accuracy per eval, in time order.
        last: Window size.

    Returns:
        Population std of the window (fewer points if the run is shorter).
    """
    return float(np.std(np.asarray(acc)[-last:]))


def default_target(runs: dict[str, list[Run]], frac: float = TARGET_FRACTION) -> float:
    """Target accuracy of D8: ``frac`` times the best accuracy any run ever reaches.

    Args:
        runs: Run name -> runs.
        frac: Fraction of the best accuracy.

    Returns:
        The target accuracy in [0, 1].
    """
    best = [r.evals["acc"].max() for rs in runs.values() for r in rs if "acc" in r.evals]
    return frac * float(max(best))


def inclusion_counts(events: list[dict], n_clients: int) -> list[int]:
    """Aggregations each client contributed to, counting a client once per ``agg`` event.

    Decision D20 ("inclusion" fairness): Libra's ``cids`` may repeat a client that uploaded several
    times in one round, which inflates the accepted-upload count ("influence"). ``set(cids)``
    asks the other question: in how many aggregations did the client take part at all?

    Args:
        events: All events of a run.
        n_clients: N.

    Returns:
        List of length N, index = client id.
    """
    counts = [0] * n_clients
    for e in events:
        if e["kind"] == "agg":
            for cid in set(e["cids"]):
                counts[cid] += 1
    return counts


def fairness_stats(counts: list[int]) -> tuple[float, float, float]:
    """Variance, Jain's index and % never for one per-client count vector.

    Re-implemented here (same formulas as ``flsim.metrics``) because plots.py must not import
    ``flsim``; population variance and the all-zero -> 1.0 convention match ``summary.json``.

    Args:
        counts: Non-negative per-client counts x_k.

    Returns:
        ``(variance, jain, pct_never)`` with pct_never in percent.
    """
    x = np.asarray(counts, dtype=np.float64)
    denom = len(x) * float((x**2).sum())
    jain = 1.0 if denom == 0.0 else float(x.sum() ** 2 / denom)
    return float(x.var()), jain, float(100.0 * np.mean(x == 0))


def n_clients_of(run: Run) -> int | None:
    """Client count N of a run, from its summary or config (None if neither is saved).

    Args:
        run: Loaded run.

    Returns:
        N, or None when unknown.
    """
    if "participation_counts" in run.summary:
        return len(run.summary["participation_counts"])
    return run.config.get("n_clients")


def reason_counts(events: list[dict], kind: str) -> dict[str, int]:
    """Count ``halt`` or ``reject`` events per ``reason``.

    Args:
        events: All events of a run.
        kind: ``"halt"`` or ``"reject"``.

    Returns:
        Reason -> count (missing reason recorded as ``"?"``).
    """
    return dict(Counter(e.get("reason", "?") for e in events if e["kind"] == kind))


def fmt_reasons(counts: dict[str, int]) -> str:
    """Render reason counts as ``"sinkhorn:40 round_end:16"`` (descending count, then name).

    Args:
        counts: Reason -> count.

    Returns:
        Space-separated string; empty when there are no events.
    """
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return " ".join(f"{k}:{v}" for k, v in items)


def target_row(t: np.ndarray, acc: np.ndarray, target: float) -> int | None:
    """Index of the first eval row whose accuracy reaches the target.

    Args:
        t: Eval times, ascending.
        acc: Test accuracy per eval.
        target: Accuracy threshold.

    Returns:
        Row index, or None when never reached.
    """
    hit = np.flatnonzero(np.asarray(acc) >= target)
    return int(hit[0]) if hit.size else None


def config_differences(configs: list[dict]) -> list[str]:
    """Keys whose value differs across seeds, ignoring ``seed`` and ``out`` (expected to differ).

    Args:
        configs: One config dict per seed.

    Returns:
        Sorted key names.
    """
    ignore = {"seed", "out"}
    keys = set().union(*configs) - ignore if configs else set()
    return sorted(k for k in keys if len({json.dumps(c.get(k)) for c in configs}) > 1)


def family(name: str) -> str:
    """Map a run name to its algorithm family for colouring.

    Args:
        name: Run name such as ``seafl_scaled`` or ``v2_libra``.

    Returns:
        One of ``fedavg``, ``seafl``, ``libra``, ``other``.
    """
    low = name.lower()
    for fam in ("fedavg", "seafl", "libra"):
        if fam in low:
            return fam
    return "other"


def colour_map(names: list[str]) -> dict[str, str]:
    """Assign each run a fixed colour: family hue, with shade by position inside the family.

    Args:
        names: All run names of the experiment.

    Returns:
        Run name -> hex colour. Deterministic because names are sorted first.
    """
    seen: dict[str, int] = {}
    out = {}
    for n in sorted(names):
        fam = family(n)
        shades = FAMILY_COLOURS[fam]
        out[n] = shades[seen.get(fam, 0) % len(shades)]
        seen[fam] = seen.get(fam, 0) + 1
    return out


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------


def load_run(path: Path, seed: str) -> Run | None:
    """Load one run directory; returns None when it has no ``evals.csv``.

    Args:
        path: Directory holding the run's files.
        seed: Seed label to store on the result.

    Returns:
        The loaded run, or None if there is nothing to plot.
    """
    evals_path = path / "evals.csv"
    if not evals_path.exists():
        return None
    with open(evals_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    run = Run(seed=seed)
    run.evals = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    run.evals["n_msgs"] = run.evals["n_up"] + run.evals["n_down"] + run.evals["n_ctrl"]
    ev_path = path / "events.jsonl"
    if ev_path.exists():
        run.events = [json.loads(ln) for ln in ev_path.read_text().splitlines() if ln.strip()]
    sm_path = path / "summary.json"
    if sm_path.exists():
        run.summary = json.loads(sm_path.read_text())
    cfg_path = path / "config.json"
    if cfg_path.exists():
        run.config = json.loads(cfg_path.read_text())
    return run


def load_experiment(root: Path) -> dict[str, list[Run]]:
    """Load every run under an experiment directory.

    Supports both ``<root>/<run_name>/<seed>/evals.csv`` and ``<root>/<run_name>/evals.csv``
    (one implicit seed, as written by the smoke runs).

    Args:
        root: ``results/<exp>``.

    Returns:
        Run name -> list of runs (one per seed), runs without data omitted.
    """
    out: dict[str, list[Run]] = {}

    def subdirs(d: Path) -> list[Path]:
        # Skip dot-dirs: Jupyter on the GPU box drops .ipynb_checkpoints into run folders.
        return sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))

    for name_dir in (p for p in subdirs(root) if p.name != "figures"):
        direct = load_run(name_dir, "0")
        if direct is not None:
            out[name_dir.name] = [direct]
            continue
        runs = [load_run(s, s.name) for s in subdirs(name_dir)]
        runs = [r for r in runs if r is not None]
        if runs:
            out[name_dir.name] = runs
    return out


# ----------------------------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------------------------


def save_fig(fig: plt.Figure, out: Path, stem: str) -> list[Path]:
    """Save a figure as PNG (dpi 200) and PDF (vector, for the IEEE report).

    Args:
        fig: Figure to save; closed afterwards.
        out: Figures directory.
        stem: File name without extension.

    Returns:
        The written paths.
    """
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def save_table(
    header: list[str], rows: list[list[str]], out: Path, stem: str, title: str
) -> list[Path]:
    """Write a table as CSV and Markdown.

    Args:
        header: Column names.
        rows: Pre-formatted cells.
        out: Figures directory.
        stem: File name without extension.
        title: Markdown heading.

    Returns:
        The written paths.
    """
    csv_path, md_path = out / f"{stem}.csv", out / f"{stem}.md"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |"]
    lines.append("|" + "---|" * len(header))
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    md_path.write_text("\n".join(lines) + "\n")
    return [csv_path, md_path]


def pm(values: list[float], digits: int = 3) -> str:
    """Format ``mean ± std`` over seeds.

    Args:
        values: One value per seed.
        digits: Decimal places.

    Returns:
        ``"0.812 ± 0.004"``.
    """
    return f"{np.mean(values):.{digits}f} ± {np.std(values):.{digits}f}"


def new_axes(xlabel: str, ylabel: str, height: float = 2.4) -> tuple[plt.Figure, plt.Axes]:
    """Create a single-column figure with labelled axes.

    Args:
        xlabel: X label including units.
        ylabel: Y label including units.
        height: Figure height in inches.

    Returns:
        ``(fig, ax)``.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, height))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return fig, ax


def legend_below(ax: plt.Axes) -> None:
    """Put the legend under the axes so it never covers data.

    Args:
        ax: Axes whose legend to place.
    """
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7, frameon=False)


# ----------------------------------------------------------------------------------------------
# Figures and tables
# ----------------------------------------------------------------------------------------------


def plot_curves(
    runs: dict[str, list[Run]],
    colours: dict[str, str],
    xkey: str,
    xlabel: str,
    out: Path,
    stem: str,
) -> list[Path]:
    """Accuracy against time or message count: mean over seeds with a +-std band.

    Args:
        runs: Run name -> runs.
        colours: Run name -> colour.
        xkey: Column of ``Run.evals`` for the x axis (``t`` or ``n_msgs``).
        xlabel: X label with units.
        out: Figures directory.
        stem: Output file stem.

    Returns:
        Written paths.
    """
    fig, ax = new_axes(xlabel, "Test accuracy (fraction)")
    for name, rs in runs.items():
        # Grid ends at the shortest seed so every grid point has data from all seeds.
        grid = np.linspace(0, min(r.evals[xkey].max() for r in rs), GRID_POINTS)
        mean, std = mean_std([interp_to_grid(r.evals[xkey], r.evals["acc"], grid) for r in rs])
        ax.plot(grid, mean, color=colours[name], label=f"{name} (n={len(rs)})", lw=1.3)
        if len(rs) > 1:
            ax.fill_between(grid, mean - std, mean + std, color=colours[name], alpha=0.2, lw=0)
    legend_below(ax)
    return save_fig(fig, out, stem)


def plot_time_to_target(
    runs: dict[str, list[Run]], colours: dict[str, str], target: float, out: Path
) -> list[Path]:
    """Bar chart of mean time-to-target; unreached runs get a hatched "not reached" bar.

    Args:
        runs: Run name -> runs.
        colours: Run name -> colour.
        target: Target accuracy in [0, 1].
        out: Figures directory.

    Returns:
        Written paths.
    """
    fig, ax = new_axes("", f"Time to {target:.0%} accuracy (s)", height=2.6)
    names = list(runs)
    ttt = {
        n: [time_to_target(r.evals["t"], r.evals["acc"], target) for r in rs]
        for n, rs in runs.items()
    }
    # The hatched bar spans the largest budget seen, so "not reached" is visibly the worst.
    budget = max(r.evals["t"].max() for rs in runs.values() for r in rs)
    for i, n in enumerate(names):
        reached = [v for v in ttt[n] if v is not None]
        if len(reached) == len(ttt[n]):
            ax.bar(i, np.mean(reached), yerr=np.std(reached), color=colours[n], capsize=2)
            ax.text(
                i, np.mean(reached), f"{np.mean(reached):.0f}", ha="center", va="bottom", fontsize=7
            )
        else:
            ax.bar(i, budget, color="white", edgecolor=colours[n], hatch="//")
            ax.text(i, budget / 2, "not\nreached", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=7)
    return save_fig(fig, out, "time_to_target")


def overhead_table(runs: dict[str, list[Run]], out: Path) -> list[Path]:
    """Communication overhead per run, normalised to FedAvg when a fedavg run exists.

    Args:
        runs: Run name -> runs.
        out: Figures directory.

    Returns:
        Written paths.
    """
    keys = ["n_up", "n_down", "n_ctrl", "total_bytes", "n_rejects", "n_halts"]
    mean = {
        n: {k: np.mean([r.summary.get(k, 0) for r in rs]) for k in keys} for n, rs in runs.items()
    }
    base = next((n for n in runs if family(n) == "fedavg"), None)
    rows = []
    for n, m in mean.items():
        cells = []
        for k in keys:
            if base is None:
                cells.append(f"{m[k]:.0f}")
            elif mean[base][k] == 0:
                cells.append(f"{m[k]:.0f} (base 0)")  # ratio undefined, show the raw value
            else:
                cells.append(f"{m[k] / mean[base][k]:.2f}x")
        rows.append([n, *cells])
    title = "Overhead" + (f" normalised to {base}" if base else " (raw means, no fedavg run)")
    return save_table(["run", *keys], rows, out, "overhead", title)


def hist_figure(
    counts: dict[str, np.ndarray], colours: dict[str, str], xlabel: str, out: Path, stem: str
) -> list[Path]:
    """Stacked per-run histograms of per-client counts (pooled over seeds).

    Args:
        counts: Run name -> per-client counts, concatenated over seeds.
        colours: Run name -> colour.
        xlabel: X label with units.
        out: Figures directory.
        stem: Output file stem.

    Returns:
        Written paths.
    """
    top = int(max(c.max() for c in counts.values()))
    bins = np.arange(top + 2) - 0.5  # integer-centred bins
    fig, axes = plt.subplots(
        len(counts), 1, figsize=(FIG_W, 1.1 * len(counts) + 0.8), sharex=True, squeeze=False
    )
    for ax, (n, c) in zip(axes[:, 0], counts.items()):
        ax.hist(c, bins=bins, color=colours[n], edgecolor="white")
        ax.set_ylabel("Clients", fontsize=7)
        ax.set_title(n, fontsize=8, loc="right", pad=1)
    axes[-1, 0].set_xlabel(xlabel)
    return save_fig(fig, out, stem)


def per_run_counts(runs: dict[str, list[Run]], kind: str) -> dict[str, list[list[int]]]:
    """Per-seed per-client count vectors for both fairness definitions (decision D20).

    Args:
        runs: Run name -> runs.
        kind: ``"influence"`` (accepted uploads, from ``summary.json``) or ``"inclusion"``
            (aggregations joined, from ``events.jsonl``).

    Returns:
        Run name -> one count vector per seed that has the needed inputs; runs without any are
        omitted.
    """
    out: dict[str, list[list[int]]] = {}
    for n, rs in runs.items():
        vecs = []
        for r in rs:
            if kind == "influence" and "participation_counts" in r.summary:
                vecs.append(list(r.summary["participation_counts"]))
            elif kind == "inclusion" and r.events and n_clients_of(r):
                vecs.append(inclusion_counts(r.events, n_clients_of(r)))
        if vecs:
            out[n] = vecs
    return out


def plot_participation(
    runs: dict[str, list[Run]], colours: dict[str, str], out: Path
) -> list[Path]:
    """Participation histograms (influence and inclusion) plus the two-definition fairness table.

    Decision D20: "influence" = accepted uploads per client (Libra's P_i, repeats count),
    "inclusion" = aggregations the client took part in (each client once per agg event). The
    table reports variance, Jain's index and % never for both, mean +- std over seeds.

    Args:
        runs: Run name -> runs.
        colours: Run name -> colour.
        out: Figures directory.

    Returns:
        Written paths.
    """
    vecs = {k: per_run_counts(runs, k) for k in ("influence", "inclusion")}
    paths: list[Path] = []
    labels = {
        "influence": ("participation_hist", "Accepted uploads per client (count)"),
        "inclusion": ("inclusion_hist", "Aggregations joined per client (count)"),
    }
    for k, (stem, xlabel) in labels.items():
        if not vecs[k]:
            print(f"skip {stem}: no {k} inputs (summary.json / events.jsonl) in any run")
            continue
        pooled = {n: np.concatenate(v) for n, v in vecs[k].items()}
        paths += hist_figure(pooled, colours, xlabel, out, stem)
    names = [n for n in runs if any(n in vecs[k] for k in vecs)]
    if not names:
        return paths
    rows = []
    for n in names:
        cells = [n]
        for k in ("influence", "inclusion"):
            stats = [fairness_stats(v) for v in vecs[k].get(n, [])]
            for i, digits in enumerate((2, 3, 1)):
                cells.append(pm([s[i] for s in stats], digits) if stats else "")
        rows.append(cells)
    header = ["run"] + [
        f"{k}: {m}" for k in ("influence", "inclusion") for m in ("variance", "Jain", "% never")
    ]
    return paths + save_table(
        header, rows, out, "fairness", "Fairness (D20: influence vs inclusion)"
    )


def plot_staleness(runs: dict[str, list[Run]], colours: dict[str, str], out: Path) -> list[Path]:
    """Box plot of the staleness of every aggregated update, pooled over seeds.

    Args:
        runs: Run name -> runs.
        colours: Run name -> colour.
        out: Figures directory.

    Returns:
        Written paths (empty if no event carries ``staleness``).
    """
    data = {
        n: [s for r in rs for e in r.events if e["kind"] == "agg" for s in e.get("staleness", [])]
        for n, rs in runs.items()
    }
    data = {n: v for n, v in data.items() if v}
    if not data:
        print("skip staleness: no agg event has a 'staleness' field")
        return []
    fig, ax = new_axes("", "Staleness of aggregated update (versions)")
    box = ax.boxplot(list(data.values()), patch_artist=True, tick_labels=list(data))
    for patch, n in zip(box["boxes"], data):
        patch.set_facecolor(colours[n])
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    return save_fig(fig, out, "staleness")


def summary_table(runs: dict[str, list[Run]], out: Path) -> list[Path]:
    """Best / final accuracy (mean +- std over seeds) and last-5-eval fluctuation.

    Args:
        runs: Run name -> runs.
        out: Figures directory.

    Returns:
        Written paths.
    """
    rows = [
        [
            n,
            str(len(rs)),
            pm([r.evals["acc"].max() for r in rs]),
            pm([r.evals["acc"][-1] for r in rs]),
            pm([fluctuation(r.evals["acc"]) for r in rs], 4),
        ]
        for n, rs in runs.items()
    ]
    return save_table(
        ["run", "seeds", "best acc", "final acc", f"fluctuation (std last {LAST_K} evals)"],
        rows,
        out,
        "summary",
        "Accuracy summary",
    )


def time_to_target_table(runs: dict[str, list[Run]], target: float, out: Path) -> list[Path]:
    """Per run: time, messages and MB at the first eval that reaches the target (D8).

    Seeds that never reach the target are excluded from the means and counted in ``reached``.

    Args:
        runs: Run name -> runs.
        target: Target accuracy in [0, 1].
        out: Figures directory.

    Returns:
        Written paths.
    """
    rows = []
    for n, rs in runs.items():
        hits = []
        for r in rs:
            i = target_row(r.evals["t"], r.evals["acc"], target)
            if i is not None:
                e = r.evals
                msgs = e["n_up"][i] + e["n_down"][i] + e["n_ctrl"][i]
                hits.append((e["t"][i], msgs, (e["bytes_up"][i] + e["bytes_down"][i]) / 1e6))
        cells = [pm([h[j] for h in hits], d) if hits else "-" for j, d in enumerate((1, 0, 2))]
        rows.append([n, f"{len(hits)}/{len(rs)}", *cells])
    return save_table(
        ["run", "reached", "time to target (s)", "messages to target", "MB to target"],
        rows,
        out,
        "time_to_target_table",
        f"Time to target accuracy {target:.3f} (reached seeds only)",
    )


# Config keys per table column group; blank cells where the algorithm does not use a key.
COMMON_KEYS = [
    "algo", "preset", "n_clients", "dataset", "dirichlet_beta", "epochs", "batch_size", "lr",
    "base_delay", "time_scale", "budget_s", "eval_every_s",
]  # fmt: skip
ALGO_KEYS = {
    "seafl": [
        "K", "stale_bound", "alpha", "mu", "theta", "n_active", "partial", "uniform_weights",
    ],
    "libra": [
        "C", "tau_max", "sigma_max", "M", "good_frac", "resource_noise", "deadline_s",
        "sinkhorn_L", "select_every", "no_limit", "random_select", "staleness_select",
    ],
    "fedavg": ["clients_per_round"],
}  # fmt: skip


def hyperparam_table(runs: dict[str, list[Run]], out: Path) -> list[Path]:
    """Hyper-parameters per run name from seed 0's ``config.json``, flagging seed-to-seed drift.

    Args:
        runs: Run name -> runs.
        out: Figures directory.

    Returns:
        Written paths (empty if no run saved a config).
    """
    # ponytail: fixed key list, ceiling: new config flags need adding by hand, upgrade path: dump
    # every config key if the table ever needs to be exhaustive.
    all_keys = COMMON_KEYS + [k for ks in ALGO_KEYS.values() for k in ks]
    rows = []
    for n, rs in runs.items():
        cfgs = [r.config for r in rs if r.config]
        if not cfgs:
            continue
        # ponytail: seed 0's config stands for the run, ceiling: differing keys are only flagged,
        # upgrade path: one row per distinct config if sweeps ever vary a key inside a run name.
        first = next((r.config for r in rs if r.seed == "0" and r.config), cfgs[0])
        algo = str(first.get("algo", ""))
        used = set(COMMON_KEYS) | set(ALGO_KEYS.get(algo, []))
        cells = [str(first.get(k, "")) if k in used else "" for k in all_keys]
        diff = [k for k in config_differences(cfgs) if k in used or k not in all_keys]
        rows.append([n, *cells, ", ".join(diff)])
    if not rows:
        print("skip hyperparams: no config.json in any run")
        return []
    return save_table(
        ["run", *all_keys, "differs across seeds"], rows, out, "hyperparams", "Hyper-parameters"
    )


def per_seed_table(runs: dict[str, list[Run]], target: float, out: Path) -> list[Path]:
    """One row per (run, seed) with every raw number, for the report writer's own statistics.

    Args:
        runs: Run name -> runs.
        target: Target accuracy in [0, 1].
        out: Figures directory.

    Returns:
        Written paths.
    """
    header = [
        "run", "seed", "best_acc", "final_acc", "time_to_target_s", "n_up", "n_down", "n_ctrl",
        "total_MB", "n_rejects", "n_halts", "jain_influence", "jain_inclusion",
        "pct_never_influence", "pct_never_inclusion", "n_aggs", "mean_staleness", "halt_reasons",
    ]  # fmt: skip
    rows = []
    for n, rs in runs.items():
        for r in sorted(rs, key=lambda r: r.seed):
            sm, ev = r.summary, r.evals
            ttt = time_to_target(ev["t"], ev["acc"], target)
            nc = n_clients_of(r)
            infl = sm.get("participation_counts")
            incl = inclusion_counts(r.events, nc) if r.events and nc else None
            fi = fairness_stats(infl) if infl else None
            fn = fairness_stats(incl) if incl else None
            stale = [s for e in r.events if e["kind"] == "agg" for s in e.get("staleness", [])]
            mb = sm["total_bytes"] / 1e6 if "total_bytes" in sm else ""
            rows.append(
                [
                    n, r.seed, f"{ev['acc'].max():.4f}", f"{ev['acc'][-1]:.4f}",
                    "" if ttt is None else f"{ttt:.1f}",
                    *(str(sm.get(k, "")) for k in ("n_up", "n_down", "n_ctrl")),
                    mb if mb == "" else f"{mb:.2f}",
                    str(sm.get("n_rejects", "")), str(sm.get("n_halts", "")),
                    f"{fi[1]:.3f}" if fi else "", f"{fn[1]:.3f}" if fn else "",
                    f"{fi[2]:.1f}" if fi else "", f"{fn[2]:.1f}" if fn else "",
                    str(sum(e["kind"] == "agg" for e in r.events)) if r.events else "",
                    f"{np.mean(stale):.3f}" if stale else "",
                    fmt_reasons(reason_counts(r.events, "halt")),
                ]
            )  # fmt: skip
    return save_table(header, rows, out, "per_seed", "Raw per-seed results")


def libra_mechanism_table(runs: dict[str, list[Run]], out: Path) -> list[Path]:
    """Per seed Libra mechanism counts: halts and rejects per reason, Sinkhorn checks, refreshes.

    Only runs whose config says ``algo == "libra"`` appear. Reason columns are the union over all
    those runs so rows line up.

    Args:
        runs: Run name -> runs.
        out: Figures directory.

    Returns:
        Written paths (empty if there is no libra run with events).
    """
    libra = [
        (n, r)
        for n, rs in runs.items()
        for r in sorted(rs, key=lambda r: r.seed)
        if r.config.get("algo") == "libra" and r.events
    ]
    if not libra:
        print("skip libra mechanisms: no libra run with config.json and events.jsonl")
        return []
    halts = [reason_counts(r.events, "halt") for _, r in libra]
    rejects = [reason_counts(r.events, "reject") for _, r in libra]
    h_cols = sorted({k for d in halts for k in d})
    r_cols = sorted({k for d in rejects for k in d})
    header = [
        "run", "seed", *(f"halt:{k}" for k in h_cols), *(f"reject:{k}" for k in r_cols),
        "sinkhorn checks", "median sinkhorn dist", "selection refreshes",
    ]  # fmt: skip
    rows = []
    for (n, r), h, rj in zip(libra, halts, rejects):
        dist = [e["dist"] for e in r.events if e["kind"] == "sinkhorn"]
        rows.append(
            [
                n, r.seed, *(str(h.get(k, 0)) for k in h_cols),
                *(str(rj.get(k, 0)) for k in r_cols), str(len(dist)),
                f"{np.median(dist):.3f}" if dist else "",
                str(sum(e["kind"] == "select" for e in r.events)),
            ]
        )  # fmt: skip
    return save_table(header, rows, out, "libra_mechanisms", "Libra mechanism counts per seed")


def write_meta(runs: dict[str, list[Run]], target: float, out: Path) -> Path:
    """Record the target actually used, generation time and the runs/seeds behind the outputs.

    Args:
        runs: Run name -> runs.
        target: Target accuracy used.
        out: Figures directory.

    Returns:
        Path of ``meta.json``.
    """
    meta = {
        "target": target,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": {n: sorted(r.seed for r in rs) for n, rs in runs.items()},
    }
    path = out / "meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    """Generate every figure and table whose inputs exist.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success, 1 if no runs were found).
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path, help="results/<exp> directory")
    ap.add_argument(
        "--target", type=float, default=None, help="accuracy target (default 90%% of best)"
    )
    args = ap.parse_args(argv)

    runs = load_experiment(args.root)
    if not runs:
        print(f"no runs with evals.csv under {args.root}")
        return 1
    out = args.root / "figures"
    out.mkdir(exist_ok=True)
    colours = colour_map(list(runs))
    target = args.target if args.target is not None else default_target(runs)
    src = "from --target" if args.target is not None else f"default {TARGET_FRACTION:.0%} of best"
    print(f"target accuracy used: {target:.3f} ({src})")  # first line: the report quotes it
    print(f"runs: {', '.join(runs)}")

    written = []
    written += plot_curves(runs, colours, "t", "Wall-clock time (s)", out, "acc_vs_time")
    written += plot_curves(
        runs, colours, "n_msgs", "Cumulative messages (count)", out, "acc_vs_msgs"
    )
    written += plot_time_to_target(runs, colours, target, out)
    written += overhead_table(runs, out)
    written += plot_participation(runs, colours, out)
    written += plot_staleness(runs, colours, out)
    written += summary_table(runs, out)
    written += time_to_target_table(runs, target, out)
    written += hyperparam_table(runs, out)
    written += per_seed_table(runs, target, out)
    written += libra_mechanism_table(runs, out)
    written.append(write_meta(runs, target, out))
    for p in written:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
