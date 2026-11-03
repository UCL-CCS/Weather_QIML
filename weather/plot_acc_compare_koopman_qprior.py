"""
Compare ACC (and optionally RMSE) vs lead time for QIML and the matched Koopman baseline.

  python plot_acc_compare_koopman_qprior.py \\
    --json-a results/baseline_results_koopman1x4ar_z500_no_qprior.json \\
    --json-b results/baseline_results_koopman1x4ar_z500_with_qprior.json \\
    --variable z500 --min-lead 24 --max-lead 240 --show-rmse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PAPER_COLORS = {
    "QIML": "#003B6F",
    "QIML (ours)": "#003B6F",
    "Koopman": "#D55E00",
}
DEFAULT_COLORS = ["#003B6F", "#D55E00", "#56B4E9", "#7E2F8E", "#009E73", "#CC79A7"]
DEFAULT_MARKERS = ["o", "s", "^", "D", "v", "P"]
DEFAULT_LINESTYLES = ["-", "-", "-", "-", "-", "-"]


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )


def color_for(label: str, index: int) -> str:
    return PAPER_COLORS.get(label, PAPER_COLORS.get(label.strip(), DEFAULT_COLORS[index % len(DEFAULT_COLORS)]))


def style_curve_ax(ax) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def legend_label_qiml_ours(s: str) -> str:
    if s.strip().upper() == "QIML":
        return "QIML (ours)"
    return s


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metric_series(data: dict, metric_key: str) -> list[tuple[int, float]]:
    wb2 = data.get("wb2_evaluation", {})
    series: list[tuple[int, float]] = []
    for _, entry in wb2.items():
        if not isinstance(entry, dict):
            continue
        if "lead_hours" not in entry or metric_key not in entry:
            continue
        v = entry[metric_key]
        if v is None:
            continue
        series.append((int(entry["lead_hours"]), float(v)))

    series.sort(key=lambda x: x[0])
    return series


def default_label(path: Path, data: dict, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    qe = data.get("qprior_enabled")
    if qe is True:
        return "With Q-Prior"
    if qe is False:
        return "No Q-Prior"
    return path.stem


def align_leads(
    a: list[tuple[int, float]],
    b: list[tuple[int, float]],
    min_lead: int | None,
    max_lead: int | None,
) -> tuple[list[int], list[float], list[float]]:
    ma = {lh: v for lh, v in a}
    mb = {lh: v for lh, v in b}
    common = sorted(set(ma.keys()) & set(mb.keys()))
    if min_lead is not None:
        common = [x for x in common if x >= min_lead]
    if max_lead is not None:
        common = [x for x in common if x <= max_lead]
    va = [ma[h] for h in common]
    vb = [mb[h] for h in common]
    return common, va, vb


def align_leads_many(
    series_list: list[list[tuple[int, float]]],
    min_lead: int | None,
    max_lead: int | None,
) -> tuple[list[int], list[list[float]]]:
    if len(series_list) < 2:
        raise ValueError("need at least two series")
    maps = [{lh: v for lh, v in s} for s in series_list]
    common: set[int] = set(maps[0].keys())
    for m in maps[1:]:
        common &= set(m.keys())
    common_sorted = sorted(common)
    if min_lead is not None:
        common_sorted = [x for x in common_sorted if x >= min_lead]
    if max_lead is not None:
        common_sorted = [x for x in common_sorted if x <= max_lead]
    vals = [[m[h] for h in common_sorted] for m in maps]
    return common_sorted, vals


def resolve_path(script_dir: Path, p: str) -> Path:
    path = Path(p)
    if not path.is_absolute() and not path.is_file():
        path = script_dir / p
    return path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Compare WB2 ACC from results JSON (2-way with gain, or multi-curve).")
    p.add_argument("--json-a", type=str, default="", help="Reference JSON (two-way mode).")
    p.add_argument("--json-b", type=str, default="", help="Comparison JSON (two-way mode).")
    p.add_argument(
        "--json",
        action="append",
        default=None,
        dest="json_multi",
        metavar="PATH",
        help="Results JSON; repeat for 3+ curves (multi mode). Two-way gain axis is disabled.",
    )
    p.add_argument("--variable", type=str, default="z500", help="Field suffix, e.g. z500 -> acc_z500")
    p.add_argument("--label-a", type=str, default="", help="Legend A (two-way).")
    p.add_argument("--label-b", type=str, default="", help="Legend B (two-way).")
    p.add_argument(
        "--labels",
        type=str,
        default="",
        help="Comma-separated legends for multi mode (count must match --json).",
    )
    p.add_argument("--min-lead", type=int, default=24, help="Minimum lead hour (inclusive)")
    p.add_argument("--max-lead", type=int, default=480, help="Maximum lead hour (inclusive)")
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path (default under plots/).",
    )
    p.add_argument(
        "--show-rmse",
        action="store_true",
        help="Add a panel below: RMSE vs lead (same lead alignment).",
    )
    args = p.parse_args()

    multi_paths: list[Path] | None = None
    if args.json_multi:
        multi_paths = [resolve_path(script_dir, x) for x in args.json_multi]
        for path in multi_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing JSON: {path}")
    elif args.json_a and args.json_b:
        path_a = resolve_path(script_dir, args.json_a)
        path_b = resolve_path(script_dir, args.json_b)
        if not path_a.is_file() or not path_b.is_file():
            raise FileNotFoundError(f"Missing JSON: {path_a} or {path_b}")
        multi_paths = [path_a, path_b]
    else:
        raise SystemExit("Either pass --json-a and --json-b, or repeat --json for each file (>=2).")

    assert multi_paths is not None
    n = len(multi_paths)
    datas = [load_json(path) for path in multi_paths]

    acc_key = f"acc_{args.variable}"
    rmse_key = f"rmse_{args.variable}"

    series_acc = [load_metric_series(d, acc_key) for d in datas]
    series_rmse = [load_metric_series(d, rmse_key) for d in datas]

    for i, path in enumerate(multi_paths):
        if not series_acc[i]:
            raise RuntimeError(f"Empty {acc_key} in {path}")

    labels: list[str]
    two_way_gain = n == 2 and args.json_multi is None and bool(args.json_a) and bool(args.json_b)
    if n == 2 and not args.json_multi:
        labels = [
            default_label(multi_paths[0], datas[0], args.label_a),
            default_label(multi_paths[1], datas[1], args.label_b),
        ]
    else:
        raw = [s.strip() for s in args.labels.split(",") if s.strip()]
        if len(raw) != n:
            raise SystemExit(f"Multi mode: need {n} comma-separated --labels, got {len(raw)}: {raw!r}")
        labels = raw

    labels = [legend_label_qiml_ours(lb) for lb in labels]

    if n == 2:
        leads, va, vb = align_leads(series_acc[0], series_acc[1], args.min_lead, args.max_lead)
        acc_cols = [va, vb]
    else:
        leads, acc_cols = align_leads_many(series_acc, args.min_lead, args.max_lead)

    if not leads:
        raise RuntimeError("No overlapping lead_hours in range after --min-lead/--max-lead.")

    imp_pct: list[float] | None = None
    if two_way_gain:
        imp_pct = []
        for v0, v1 in zip(acc_cols[0], acc_cols[1]):
            if v0 == 0.0:
                imp_pct.append(float("nan"))
            else:
                imp_pct.append((v1 - v0) / v0 * 100.0)

    plots_dir = script_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if args.output.strip():
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = script_dir / out_path
    else:
        stem = "_vs_".join(path.stem for path in multi_paths)
        if len(stem) > 120:
            stem = f"{n}models_{args.variable}"
        suffix = "_rmse" if args.show_rmse else ""
        out_path = plots_dir / f"acc_compare_{stem}{suffix}.png"

    apply_paper_style()
    nrows = 2 if args.show_rmse else 1
    fig_h = 4.8 if nrows == 1 else 8.0
    fig, ax_arr = plt.subplots(
        nrows,
        1,
        figsize=(8.2, fig_h),
        sharex=True,
        constrained_layout=True,
    )
    ax = np.atleast_1d(ax_arr).ravel()[0]

    def _plot_curves(ax_i, x, cols, *, highlight_qiml: bool = True):
        lines_local = []
        for i in range(n):
            c = color_for(labels[i], i)
            m = DEFAULT_MARKERS[i % len(DEFAULT_MARKERS)]
            is_qiml = "QIML" in labels[i].upper()
            lw = 2.8 if (highlight_qiml and is_qiml) else 2.0
            z = 5 if (highlight_qiml and is_qiml) else 3
            (ln,) = ax_i.plot(
                x,
                cols[i],
                color=c,
                linestyle="-",
                linewidth=lw,
                marker=m,
                markersize=6.5 if is_qiml else 5.5,
                markerfacecolor=c,
                markeredgecolor="white",
                markeredgewidth=0.9,
                label=labels[i],
                zorder=z,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            lines_local.append(ln)
        style_curve_ax(ax_i)
        return lines_local

    lines = _plot_curves(ax, leads, acc_cols)
    ax.set_ylabel(f"ACC ({args.variable})")
    legend_handles = list(lines)
    legend_labels = list(labels)

    if two_way_gain and imp_pct is not None:
        ax2 = ax.twinx()
        ax2.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--", alpha=0.7)
        (line_imp,) = ax2.plot(
            leads,
            imp_pct,
            marker="D",
            markersize=5.5,
            markerfacecolor="#009E73",
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=1.8,
            linestyle="--",
            color="#009E73",
            label="Relative ACC gain (B vs A) [%]",
            zorder=4,
        )
        ax2.set_ylabel("Relative ACC gain (B − A) / A × 100 [%]", color="#009E73")
        ax2.tick_params(axis="y", labelcolor="#009E73")
        ax2.spines["top"].set_visible(False)
        legend_handles = [lines[0], lines[1], line_imp]
        legend_labels = [labels[0], labels[1], "Relative ACC gain [%]"]

    ax.legend(legend_handles, legend_labels, loc="upper right", borderaxespad=0.2)
    var_u = args.variable.upper()
    ax.set_title(f"{var_u} forecast, {args.min_lead}–{args.max_lead} h", pad=8)

    if args.show_rmse:
        for s in series_rmse:
            if not s:
                raise RuntimeError(f"--show-rmse requires non-empty {rmse_key} in every JSON.")
        if n == 2:
            leads_r, r0, r1 = align_leads(series_rmse[0], series_rmse[1], args.min_lead, args.max_lead)
            rmse_cols = [r0, r1]
        else:
            leads_r, rmse_cols = align_leads_many(series_rmse, args.min_lead, args.max_lead)
        if leads_r != leads:
            raise RuntimeError("RMSE and ACC lead sets differ after alignment.")
        ax_rmse = np.atleast_1d(ax_arr).ravel()[1]
        _plot_curves(ax_rmse, leads_r, rmse_cols)
        ax_rmse.set_ylabel(f"RMSE ({args.variable})")
        ax_rmse.set_title(f"{var_u} RMSE, {args.min_lead}–{args.max_lead} h", pad=8)
        ax_rmse.legend(loc="upper left", borderaxespad=0.2)
        ax_rmse.set_xlabel("Lead time (hours)")
    else:
        ax.set_xlabel("Lead time (hours)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {pdf_path}")
    print(f"Common lead hours: {len(leads)} points, {leads[0]}-{leads[-1]} h")


if __name__ == "__main__":
    main()
