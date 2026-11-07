# qmlbench: where do quantum regression models fit?

A fair benchmark of **quantum vs classical regression** on four real-world domains, chosen to
test one hypothesis: quantum models (sums of Fourier modes in their rotation-angle inputs)
should have the most natural fit on data with **genuinely periodic inputs** and in
**small-data** regimes.

> **▶ [Try the interactive demo](https://tuananhbuisieucapdeptrai.github.io/quantum-project/demo/)**:
> can a quantum computer predict the tide? Train a quantum-kernel model in your browser; no
> quantum background needed. &nbsp;·&nbsp; **[Read the paper (PDF)](report/main.pdf)**

| Domain | Why it's here |
|---|---|
| 🌊 **Tides** (NOAA, San Francisco) | Inputs are tidal phases; classical harmonic analysis is itself a Fourier series |
| 💨 **Wind** (Kelmarsh wind farm) | Wind direction is a natural angle; wake effects make power depend on it |
| ⚽ **Soccer xG** (StatsBomb) | Small data, geometric features, probability output |
| 🔭 **Astronomy photo-z** (SDSS) | Ordinary tabular control domain: no periodicity |

Full problem statement, models and protocol: [docs/FORMULATION.md](docs/FORMULATION.md).
Author: Tuan Bui. License: MIT.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
source .venv/bin/activate
qmlbench-fetch            # downloads + preprocesses all four datasets (~600 MB transfer, ~130 MB on disk)
pytest                    # unit tests + checks on the downloaded data
python -m qmlbench.run --dry-run
```

## Running the benchmark

```bash
python -m qmlbench.run --jobs 6                 # full grid from configs/benchmark.yaml (resumable)
python -m qmlbench.run --datasets tides --models vqc qkernel domain --n-train 250 --seeds 0
python -m qmlbench.run --collect                # merge finished runs -> results/all_runs.parquet
```

Each run tunes its model with the same Optuna budget (inner CV on the training subsample,
folds that respect time order / match groups), refits, and writes one JSON file to
`results/raw/<dataset>/`. Interrupt at any time; re-running skips finished runs.

## Results

No quantum model beats the best classical or domain method on any dataset at any training size
(850 runs). The angle-map quantum kernel comes close on tides and photo-z; the IQP feature map is
rejected on periodic inputs because its x_j·x_k terms break periodicity. On tides, a Fourier
analysis shows that a one-layer circuit's frequencies already allow test R² 0.932, so the
variational circuit's shortfall comes from training and from the coefficients its ansatz can
reach, not from missing frequencies ([docs/FOURIER.md](docs/FOURIER.md)).

Test score with 1000 training points (mean of 5 seeds; R², Brier skill score for soccer):

| | Tides | Wind | Soccer | Photo-z |
|---|---|---|---|---|
| Best classical | 0.931 | 0.981 | 0.091 | 0.909 |
| Domain method | 0.930 | 0.980 | 0.066 | 0.882 |
| Quantum kernel | 0.927 | 0.953 | 0.087 | 0.902 |
| Re-uploading circuit | 0.894 | 0.976 | 0.061 | 0.893 |

- **Paper:** [report/main.pdf](report/main.pdf)
- **Interactive tides demo** (for a general audience): <https://tuananhbuisieucapdeptrai.github.io/quantum-project/demo/> (source in [demo/](demo/))

## Reproducing the paper and demo

The per-run results are in `results/raw/` (one JSON per run), so the paper and demo can be
rebuilt without re-running the benchmark:

```bash
python scripts/fourier_tides.py --probe-vqc     # spectrum ceilings and VQC probes (~3 min)
python scripts/make_report_assets.py            # every table, figure and number in the paper
cd report && tectonic main.tex                  # -> report/main.pdf
python scripts/build_tide_demo.py               # -> demo/data/tides.json (needs the tides data)
python3 -m http.server 8765                     # then open http://localhost:8765/demo/
```

The full benchmark took about 20 CPU-hours, almost all of it training the variational circuits.
The first VQC run, with a shorter training budget, is kept in `results/archive/vqc_v1/`.

## Layout

```
configs/            one YAML per dataset + benchmark.yaml (grid, tuning budget)
src/qmlbench/
  data/             loaders: tides, wind, soccer, photoz (+ io.py, base.py, fetch_all.py)
  encoding.py       AngleScaler (quantum) and PeriodicExpander (classical)
  splits.py         temporal / group / random splits, training-size subsampling
  metrics.py        RMSE, R², Brier, log-loss, sigma_NMAD, outlier fraction
  models/           classical.py, quantum_vqc.py, quantum_kernel.py, domain.py
  tuning.py         equal-budget Optuna search with structure-aware inner CV
  fourier.py        frequency ceilings and model spectra
  run.py            benchmark runner
scripts/            Fourier analysis, paper assets, demo data
results/            per-run JSONs, merged parquet, Fourier outputs
report/             paper (main.tex) and its generated tables, figures and numbers
demo/               static tides demo website
docs/               problem formulation, Fourier analysis
tests/
```

## Data sources and attribution

- **Tides:** NOAA CO-OPS Data API, station 9414290 (public domain).
- **Wind:** Kelmarsh wind farm data, Cubico Sustainable Investments, Zenodo record 8252025, CC-BY 4.0.
- **Soccer:** [StatsBomb Open Data](https://github.com/statsbomb/open-data). Data provided by StatsBomb.
- **Astronomy:** Sloan Digital Sky Survey DR18 via SkyServer. Funding for SDSS is acknowledged per
  https://www.sdss.org/collaboration/citing-sdss/.

## Key references

- Bowles, Ahmed, Schuld (2024). *Better than classical? The subtle art of benchmarking quantum machine learning models.* arXiv:2403.07059
- Schuld, Sweke, Meyer (2021). *Effect of data encoding on the expressive power of variational quantum machine learning models.* PRA 103, 032430
- Schuld (2021). *Supervised quantum machine learning models are kernel methods.* arXiv:2101.11020
- Landman et al. (2022). *Classically approximating variational quantum machine learning with random Fourier features.* arXiv:2210.13200
- Pérez-Salinas et al. (2020). *Data re-uploading for a universal quantum classifier.* Quantum 4, 226
