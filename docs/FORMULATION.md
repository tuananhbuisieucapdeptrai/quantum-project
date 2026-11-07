# Problem formulation

## 1. Research question

> **Where, if anywhere, do quantum regression models have a useful inductive bias on real data?**

Bowles, Ahmed & Schuld (2024) found that, under fair benchmarking, out-of-the-box quantum
classifiers rarely beat simple classical baselines. This project asks the same question for
**regression**, and asks it *per data type*: data whose inputs are genuinely periodic
(tidal phases, wind direction) versus ordinary tabular data (galaxy colours), and data-rich
versus data-poor settings.

| | Research question |
|---|---|
| **RQ1** | On each domain, how do quantum models compare with tuned classical and domain-expert baselines, at equal tuning budget? |
| **RQ2** | Is the gap smaller (or reversed) on periodic/angular domains than on the tabular control domain? |
| **RQ3** | How does the gap change with training-set size (50 → 1000)? |
| **RQ4** | Can Random Fourier Features reproduce the quantum models' accuracy (dequantisation)? |
| **RQ5** | Do the Fourier spectra that the quantum models can express match the spectra present in the data? |

A negative answer is a valid result. The goal is a careful map of when these models help or don't.

## 2. Learning problem

For each domain we have $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$, $x_i \in \mathbb{R}^d$, and
each feature $j$ has a type $\tau_j \in \{\text{angle}, \text{real}, \text{binary}\}$.
A model $f$ is trained on $n \ll N$ points drawn from a fixed training pool and scored on a
fixed test set chosen by a domain-appropriate split.

* **Regression** (tides, wind, photoz): minimise squared error; report RMSE, MAE, $R^2$
  (+ $\sigma_\text{NMAD}$ and outlier fraction for photo-z).
* **Probability** (soccer): $y \in \{0,1\}$, $f(x) \approx P(y=1 \mid x)$; report Brier score,
  log-loss, Brier skill vs the base rate, and calibration.

## 3. Encoding: the core of the comparison

Quantum models receive features as rotation angles via `AngleScaler`:

| type | quantum encoding | classical encoding (`PeriodicExpander`) |
|---|---|---|
| angle | unchanged, $x \in [-\pi, \pi)$, so wrap-around is exact | $(\sin x, \cos x)$ |
| real | robust min–max to $[-\pi/2, \pi/2]$ (width $\pi$, clipped) | standardised |
| binary | $\{-\pi/2, \pi/2\}$ | $\{0, 1\}$ |

Reals are mapped onto a *half* circle on purpose. On the full circle the minimum and maximum
would become the same rotation, inventing a periodicity the data doesn't have. Classical
models get sin/cos for angles so that both families know the same facts about periodicity.

## 4. Models

**Quantum (PennyLane, ≤ 8 qubits, one qubit per feature)**

1. *Data re-uploading regressor.*
   $f_\theta(x) = w\,\langle 0| U^\dagger(x,\theta)\, M\, U(x,\theta) |0\rangle + b$,
   $U = W_L S(x) \cdots W_1 S(x) W_0$, $S(x) = \bigotimes_j R_Y(x_j)$.
   By Schuld, Sweke & Meyer (2021), $f_\theta(x) = \sum_{\omega \in \Omega} c_\omega(\theta) e^{i\omega\cdot x}$
   with $\Omega = \{-L,\dots,L\}^d$: a trigonometric polynomial of degree $L$ per feature.
2. *Quantum kernel.* $k(x,x') = |\langle\phi(x)|\phi(x')\rangle|^2$ with angle or IQP feature maps,
   used in kernel ridge regression / a Gaussian process (Schuld 2021).

**Classical:** ridge / logistic regression, Random Fourier Features + ridge, MLP,
histogram gradient boosting, Gaussian process (RBF + noise).

**Domain experts:** UTide harmonic analysis and NOAA's official prediction (tides), binned
IEC-style power curve (wind), distance–angle logistic xG and StatsBomb xG (soccer),
k-nearest-neighbour photo-z (astronomy).

## 5. Protocol

* Grid: dataset × model × $n_\text{train} \in \{50, 100, 250, 500, 1000\}$ × 5 seeds.
* Equal tuning budget: 10 Optuna trials per model, 3-fold CV on the training subsample only.
* Fixed test set per domain (capped at 2000 points), identical for all models.
* Report mean ± std over seeds; each run also records parameter count and fit time.
* Splits: temporal (tides, wind), by match (soccer), random (photoz).

## 6. Domains

| Domain | Source | N | Features (type) | Target | Split |
|---|---|---|---|---|---|
| Tides | NOAA CO-OPS, San Francisco 9414290, 2021–23 hourly | 26,280 | 5 constituent phases (angle) | water level, m | temporal: test = 2023 |
| Wind | Kelmarsh turbine 1, 2017, 10-min SCADA | 51,469 | wind speed (real), direction (angle) | power / rated | temporal: test = Oct–Dec |
| Soccer | StatsBomb: WC 2022, Euro 2020/2024, Copa América 2024 | 4,709 shots | distance, goal angle (real), bearing (angle), 4 flags (binary) | goal (0/1) | by match, 25 % |
| Astronomy | SDSS DR18 galaxies, r < 19, 1/17 fibre subsample | 50,693 | 4 colours + r mag (real) | redshift | random, 25 % |

Reference numbers already measured on the test periods / full data:
NOAA official tide prediction RMSE 0.105 m; StatsBomb xG Brier 0.069 (base rate 0.084).

## 7. Threats to validity

* Simulated, noiseless circuits. Shot-noise robustness is left to future work.
* Small $n_\text{train}$ is forced by simulation cost; say so, and show the trend with $n$.
* Tides from time alone cannot capture storm surge, so there is an irreducible error floor.
* The soccer set mixes four tournaments.
* The SDSS sample is spectroscopic (biased towards bright galaxies); that's standard for photo-z training.
