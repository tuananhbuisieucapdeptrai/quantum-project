"""Build every table, figure and quoted number of the paper (report/main.tex) from the results.

    python scripts/make_report_assets.py

Reads results/raw/*/*.json (the benchmark) and results/fourier/*.csv (scripts/fourier_tides.py)
and writes report/tables/*.tex, report/figures/*.pdf and report/generated/numbers.tex.
Re-run it whenever results change; the paper never contains hand-copied numbers.

The re-uploading circuit (VQC) results are the second, longer-budget run. The first run is
kept in results/archive/vqc_v1/ and is used only for the sentence comparing the two budgets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, ARCHIVE_V1 = ROOT / "results" / "raw", ROOT / "results" / "archive" / "vqc_v1"
FOURIER = ROOT / "results" / "fourier"
REPORT = ROOT / "report"
TABLES, FIGURES, GEN = REPORT / "tables", REPORT / "figures", REPORT / "generated"

DATASETS = ["tides", "wind", "soccer", "photoz"]
DS_LABEL = {"tides": "Tides", "wind": "Wind", "soccer": "Soccer", "photoz": "Photo-$z$"}
DS_PLAIN = {"tides": "Tides", "wind": "Wind", "soccer": "Soccer", "photoz": "Photo-z"}
METRIC = {"tides": "r2", "wind": "r2", "photoz": "r2", "soccer": "brier_skill"}
METRIC_TEX = {"r2": "$R^2$", "brier_skill": "BSS"}
METRIC_PLAIN = {"r2": "$R^2$", "brier_skill": "Brier skill score"}
DOMAIN = {"tides": "domain:harmonic", "wind": "domain:power_curve",
          "soccer": "domain:xg_logistic", "photoz": "domain:knn"}
REFERENCE = {"tides": "domain:noaa", "soccer": "domain:statsbomb_xg"}
ROWS = [  # (label, model id or per-dataset dict, group)
    ("Ridge / logistic", "ridge", "Classical"),
    ("Random Fourier features", "rff", "Classical"),
    ("Multilayer perceptron", "mlp", "Classical"),
    ("Gradient boosting", "gbm", "Classical"),
    ("Gaussian process", "gp", "Classical"),
    ("Re-uploading circuit", "vqc", "Quantum"),
    ("Quantum kernel", "qkernel", "Quantum"),
    ("Domain method$^{a}$", DOMAIN, "Domain"),
    ("Published reference$^{b}$", REFERENCE, "Domain"),
]
VQC_BUDGET = (400, 40, 128)  # epochs, patience, batch size (defaults of VQCRegressor)

# Paper figure style: Computer-Modern-like serif, small type, recessive axes.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"gp": "#2a78d6", "qkernel": "#eb6834", "vqc": "#1baf7a", "domain": MUTED}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    return plt


def _axes_common(ax):
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 250, 500, 1000], ["50", "100", "250", "500", "1000"])
    ax.minorticks_off()
    ax.grid(axis="y", color=GRID, lw=0.6)


# --- data ----------------------------------------------------------------------------------

def _read(files) -> pd.DataFrame:
    return pd.DataFrame([json.loads(Path(f).read_text()) for f in files])


def load_runs() -> pd.DataFrame:
    return _read(sorted(RAW.glob("*/*.json")))


# --- tables --------------------------------------------------------------------------------

def _cell(s: pd.Series) -> tuple[float, str]:
    m, sd = s.mean(), s.std()
    if not sd > 1e-12:  # fixed predictions (published references): no spread to show
        return m, f"${m:.3f}$"
    return m, f"${m:.3f}\\pm{sd:.3f}$"


def results_table(df: pd.DataFrame, n: int, path: Path, label: str, wide: bool) -> None:
    cells, best = {}, {}
    for ds in DATASETS:
        sub = df[(df.dataset == ds) & (df.n_train == n)]
        for lab, model, group in ROWS:
            mid = model.get(ds) if isinstance(model, dict) else model
            vals = sub[sub.model == mid][METRIC[ds]] if mid else pd.Series(dtype=float)
            cells[lab, ds] = _cell(vals) if len(vals) else (np.nan, "---")
            if not lab.startswith("Published") and not np.isnan(cells[lab, ds][0]):
                best[ds] = max(best.get(ds, -np.inf), cells[lab, ds][0])
    env = "table"
    caption = (f"Test scores with $n={n}$ training points: mean $\\pm$ standard deviation over "
               "five seeds (higher is better). Bold marks the best model trained on the same "
               f"points. {METRIC_TEX['r2']} for regression; BSS is the Brier skill score "
               "relative to the base rate, for the probability task.")
    lines = [f"\\begin{{{env}}}[{'t' if wide else 'H'}]", r"\centering\small\setlength{\tabcolsep}{4.5pt}",
             f"\\caption{{{caption}}}\\label{{{label}}}",
             r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             " & Model & " + " & ".join(f"{DS_LABEL[d]} ({METRIC_TEX[METRIC[d]]})"
                                        for d in DATASETS) + r" \\", r"\midrule"]
    prev = None
    for lab, _, group in ROWS:
        if prev and group != prev:
            lines.append(r"\midrule")
        row = []
        for ds in DATASETS:
            m, txt = cells[lab, ds]
            is_best = (not np.isnan(m) and np.isclose(m, best.get(ds, np.nan))
                       and not lab.startswith("Published"))
            row.append(f"\\best{{{txt}}}" if is_best else txt)
        lines.append(f"{group if group != prev else ''} & {lab} & " + " & ".join(row) + r" \\")
        prev = group
    lines += [r"\bottomrule", r"\end{tabular}",
              (r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize "
              r"$^{a}$UTide harmonic analysis (tides), binned power curve (wind), "
              r"distance--angle logistic regression (soccer) and $k$-nearest neighbours "
              r"(photo-$z$), fitted on the same training points as every other model. "
              r"$^{b}$NOAA's official tide prediction and StatsBomb's expected-goals model; "
              r"both are fitted on far more data and mark a practical ceiling rather than a "
              r"competitor.\end{minipage}"),
              f"\\end{{{env}}}"]
    path.write_text("\n".join(lines) + "\n")


def fourier_tables() -> pd.DataFrame:
    ceil = pd.read_csv(FOURIER / "tides_ceilings.csv")
    full = ceil[ceil.seed < 0].set_index("set")
    names = {"additive, |w|<=1": (r"$\Omega_1^{\mathrm{add}}$", "single-feature terms only"),
             "full, |w_j|<=1": (r"$\Omega_1$", "one-layer circuit, angle kernel"),
             "full, |w_j|<=2": (r"$\Omega_2$", "two-layer circuit")}
    rows = [f"{sym} & {desc} & {int(full.loc[k, 'n_freqs']):,} & "
            f"{full.loc[k, 'test_r2']:.3f} \\\\" for k, (sym, desc) in names.items()]
    (TABLES / "fourier_ceilings.tex").write_text("\n".join([
        r"\begin{tabular}{@{}llrc@{}}", r"\toprule",
        r"Set & Model with this spectrum & Frequencies & Ceiling $R^2$ \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}"]) + "\n")

    probe = pd.read_csv(FOURIER / "tides_vqc_probe.csv")
    rows = []
    for _, r in probe.iterrows():
        budget = "short" if r.epochs == 100 else "long"
        p2 = r.get("power_share_order2", 0.0)
        rows.append(f"{int(r.n_layers)} & {budget} & {r.target} & {int(r.epochs_run)} & "
                    f"{r.train_r2:.3f} & {r.test_r2:.3f} & {100 * p2:.0f} \\\\")
    (TABLES / "vqc_probe.tex").write_text("\n".join([
        r"\begin{tabular}{@{}rllrccr@{}}", r"\toprule",
        r"$L$ & Budget & Target & Epochs & Train $R^2$ & Test $R^2$ & $P_2$ (\%) \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}"]) + "\n")
    return full


def _num(v: float) -> str:
    """1e-06 -> 10^{-6}; 0.05 -> 0.05 (for math mode)."""
    e = np.log10(v)
    if (v < 0.01 or v >= 1000) and np.isclose(e, round(e)):
        return f"10^{{{round(e)}}}"
    return f"{v:g}"


def search_space_table() -> None:
    """Appendix table of every tuned hyperparameter, read from the code itself."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from qmlbench.models import domain
    from qmlbench.run import SPACES
    label = {"ridge": "Ridge / logistic", "rff": "Random Fourier features",
             "mlp": "Multilayer perceptron", "gbm": "Gradient boosting",
             "gp": "Gaussian process", "vqc": "Re-uploading circuit",
             "qkernel": "Quantum kernel"}
    kind = {"log": "log", "float": "linear", "int": "integer"}

    def fmt(spec):
        if spec[0] == "cat":
            return "\\{" + ", ".join(map(str, spec[1])) + "\\}"
        lo, hi = spec[1], spec[2]
        f = _num if spec[0] != "int" else str
        return f"$[{f(lo)}, {f(hi)}]$, {kind[spec[0]]}"

    rows = []
    for key, name in label.items():
        space = SPACES[key]
        if not space:
            rows.append(f"{name} & \\multicolumn{{2}}{{l}}{{none (marginal likelihood)}} \\\\")
        for i, (p, spec) in enumerate(space.items()):
            rows.append(f"{name if i == 0 else ''} & \\texttt{{{p.replace('_', chr(92) + '_')}}} "
                        f"& {fmt(spec)} \\\\")
    for ds, models in domain.SEARCH_SPACES.items():
        for mname, space in models.items():
            for i, (p, spec) in enumerate(space.items()):
                rows.append(f"{DS_PLAIN[ds]}: {mname.replace('_', ' ')} & "
                            f"\\texttt{{{p.replace('_', chr(92) + '_')}}} & {fmt(spec)} \\\\")
    (TABLES / "search_spaces.tex").write_text("\n".join([
        r"\begin{tabular}{@{}lll@{}}", r"\toprule",
        r"Model & Hyperparameter & Range \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}"]) + "\n")


# --- figures -------------------------------------------------------------------------------

def learning_curves(df: pd.DataFrame, path: Path) -> None:
    plt = _style()
    series = [("gp", "Gaussian process", "-"), ("qkernel", "Quantum kernel", "-"),
              ("vqc", "Re-uploading circuit", "-"), ("domain", "Domain method", "--")]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6))
    for i, (ax, ds) in enumerate(zip(axes.ravel(), DATASETS)):
        sub, met = df[df.dataset == ds], METRIC[ds]
        vals = []
        for key, label, ls in series:
            mid = DOMAIN[ds] if key == "domain" else key
            g = sub[sub.model == mid].groupby("n_train")[met]
            m, s = g.mean(), g.std()
            ax.fill_between(m.index, m - s, m + s, color=COLORS[key], alpha=0.10, lw=0)
            ax.plot(m.index, m.values, color=COLORS[key], ls=ls, lw=1.2, marker="o", ms=2.5,
                    label=label)
            vals.append(m.values)
        top = max(np.nanmax(v) for v in vals)
        if ds in REFERENCE:
            ref = sub[sub.model == REFERENCE[ds]][met].mean()
            ax.axhline(ref, color=MUTED, lw=0.8, ls=":", label="Published reference")
            top = max(top, ref)
        floor = {"tides": 0.75, "wind": 0.90, "soccer": -0.20, "photoz": 0.70}[ds]
        lo = max(min(np.nanmin(v) for v in vals), floor)
        pad = 0.06 * (top - lo)
        ax.set_ylim(lo - pad, top + pad)
        _axes_common(ax)
        ax.set_title(f"({'abcd'[i]}) {DS_PLAIN[ds]}", loc="left", color=INK)
        ax.set_ylabel(f"test {METRIC_PLAIN[met]}")
        if i >= 2:
            ax.set_xlabel("training points")
    handles, labels = [], []
    for ax in axes.ravel():  # collect every series once (the reference line is not on all panels)
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), labelcolor=INK, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=1.2, w_pad=1.5)
    fig.savefig(path)
    plt.close(fig)


def tides_ceiling_figure(df: pd.DataFrame, full: pd.DataFrame, path: Path) -> None:
    plt = _style()
    runs = df[df.dataset == "tides"]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for key, ls, text in [("full, |w_j|<=1", "--", r"$\Omega_1$ ceiling"),
                          ("additive, |w|<=1", ":", r"$\Omega_1^{\mathrm{add}}$ ceiling")]:
        v = full.loc[key, "test_r2"]
        ax.axhline(v, color=MUTED, lw=0.9, ls=ls, zorder=1)
        ax.annotate(f"{text} ({v:.3f})", (1150, v), xytext=(0, 2), textcoords="offset points",
                    color=MUTED, fontsize=6.5, ha="right", va="bottom")
    for key, label in [("gp", "Gaussian process"), ("qkernel", "Quantum kernel"),
                       ("vqc", "Re-uploading circuit")]:
        g = runs[runs.model == key].groupby("n_train").r2
        m, s = g.mean(), g.std()
        ax.fill_between(m.index, m - s, m + s, color=COLORS[key], alpha=0.12, lw=0)
        ax.plot(m.index, m.values, color=COLORS[key], lw=1.3, marker="o", ms=2.8, label=label,
                zorder=3)
    _axes_common(ax)
    ax.set_xlim(42, 1200)
    ax.set_ylim(0.75, 0.95)
    ax.set_xlabel("training points")
    ax.set_ylabel(r"test $R^2$ (2023)")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


# --- numbers -------------------------------------------------------------------------------

def numbers(df: pd.DataFrame, full: pd.DataFrame) -> None:
    """Key numbers as LaTeX macros, so the prose never goes stale."""
    def m(ds, model, n=1000):
        return df[(df.dataset == ds) & (df.model == model) & (df.n_train == n)][METRIC[ds]].mean()

    def best_classical(ds, n=1000):
        sub = df[(df.dataset == ds) & (df.n_train == n) & (df.family == "classical")]
        return sub.groupby("model")[METRIC[ds]].mean().max()

    epochs, patience, batch = VQC_BUDGET
    vqc = df[df.model == "vqc"]
    macros = {
        "nRuns": f"{len(df):,}".replace(",", "{,}"),
        "vqcEpochs": str(epochs), "vqcPatience": str(patience), "vqcBatch": str(batch),
        "vqcHours": f"{(vqc.tune_seconds + vqc.fit_seconds).sum() / 3600:.1f}",
        "allHours": f"{(df.tune_seconds + df.fit_seconds).sum() / 3600:.1f}",
        "ceilAdd": f"{full.loc['additive, |w|<=1', 'test_r2']:.3f}",
        "ceilOne": f"{full.loc['full, |w_j|<=1', 'test_r2']:.3f}",
        "ceilTwo": f"{full.loc['full, |w_j|<=2', 'test_r2']:.3f}",
        "tidesNOAA": f"{m('tides', 'domain:noaa'):.3f}",
        "soccerSB": f"{m('soccer', 'domain:statsbomb_xg'):.3f}",
    }
    for ds in DATASETS:
        c = ds.capitalize()
        macros[f"{ds}GP"] = f"{m(ds, 'gp'):.3f}"
        macros[f"{ds}QK"] = f"{m(ds, 'qkernel'):.3f}"
        macros[f"{ds}VQC"] = f"{m(ds, 'vqc'):.3f}"
        macros[f"{ds}Dom"] = f"{m(ds, DOMAIN[ds]):.3f}"
        macros[f"{ds}BestCl"] = f"{best_classical(ds):.3f}"
        qk = df[(df.model == "qkernel") & (df.dataset == ds)]
        fm = qk.params.map(lambda p: json.loads(p).get("feature_map"))
        macros[f"qkAngle{c}"] = str(int((fm == "angle").sum()))
        macros[f"qkRuns{c}"] = str(len(qk))
        layers = vqc[vqc.dataset == ds].params.map(lambda p: json.loads(p).get("n_layers"))
        macros[f"vqcOneLayer{c}"] = str(int((layers == 1).sum()))
        macros[f"vqcRuns{c}"] = str(len(layers))
    # v1 (short budget) VQC, kept in the archive, for the text comparing the two budgets.
    v1 = _read(sorted(ARCHIVE_V1.glob("*/*.json")))
    if len(v1):
        t1 = v1[(v1.dataset == "tides") & (v1.n_train == 1000)]
        macros["tidesVQCvOne"] = f"{t1.r2.mean():.3f}"
        macros["tidesVQCvOneSd"] = f"{t1.r2.std():.3f}"
    t2 = vqc[(vqc.dataset == "tides") & (vqc.n_train == 1000)]
    macros["tidesVQCSd"] = f"{t2.r2.std():.3f}"
    if "n_epochs_run" in vqc and vqc.n_epochs_run.notna().any():
        macros["tidesVQCEpochs"] = f"{vqc[vqc.dataset == 'tides'].n_epochs_run.median():.0f}"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    for d in (TABLES, FIGURES, GEN):
        d.mkdir(parents=True, exist_ok=True)
    df = load_runs()
    results_table(df, 1000, TABLES / "results_n1000.tex", "tab:n1000", wide=True)
    results_table(df, 100, TABLES / "results_n100.tex", "tab:n100", wide=False)  # one-column appendix
    full = fourier_tables()
    search_space_table()
    learning_curves(df, FIGURES / "learning_curves.pdf")
    tides_ceiling_figure(df, full, FIGURES / "tides_ceiling.pdf")
    numbers(df, full)
    print(f"{len(df)} runs -> {REPORT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
