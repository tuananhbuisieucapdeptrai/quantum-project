# Fourier analysis: why the VQC stalls on tides

Reproduce: `python scripts/fourier_tides.py --probe-vqc` (≈3 min). Outputs are in `results/fourier/`.
Code: `src/qmlbench/fourier.py` (tests in `tests/test_fourier.py`).

## Question

In the first benchmark run (short training budget, archived in `results/archive/vqc_v1/`) the
tuned VQC plateaued at test R² ≈ 0.87 on tides for n ≥ 250. That is below plain ridge on sin/cos
features (0.899), and the tuner picked a 1-layer circuit in 23 of 25 runs.
A 1-layer re-uploading circuit can only use frequencies ω ∈ {−1, 0, 1}⁵ in the five constituent
phases. Our first hypothesis was that tides need higher harmonics (shallow-water overtides such as
M4 = 2·M2), so the circuit simply cannot express the signal.

## Method

1. **Frequency ceilings.** For a frequency set Ω, fit the target by least squares on
   {1, cos(ω·x), sin(ω·x) : ω ∈ Ω}, using all 17,520 training hours (2021–2022), and score it on
   2023. No model limited to Ω can beat this on average, so it is the ceiling for that spectrum.
2. **Model spectra.** Sample a fitted model on a regular (2K+1)⁵ grid of phases and take the
   5-D FFT. For a band-limited model (the VQC), this gives its Fourier coefficients exactly.
3. **VQC probes** (n = 1000, seed 0, lr 0.046, 5 qubits). We vary depth and training budget. We
   also run a *teacher* test: train on the noise-free best 1-layer Fourier fit. Here the target
   is inside the circuit's spectrum by construction, so any shortfall is about which
   coefficients the circuit can reach and how well it trains.

## Results

**Ceilings (test R², 2023):**

| Frequency set | # frequencies | Ceiling |
|---|---|---|
| Additive, \|ω\| ≤ 1 (one feature at a time = ridge on sin/cos) | 5 | 0.899 |
| Full, \|ω_j\| ≤ 1 (1-layer VQC, angle kernel) | 121 | **0.932** |
| Full, \|ω_j\| ≤ 2 (2-layer VQC) | 1,562 | 0.941 |

For reference (n = 1000, mean of 5 seeds): GP 0.931, UTide 0.930, quantum kernel 0.927; VQC 0.870
in the first run and 0.894 after the re-run described under Caveats.

**VQC probes (n = 1000):**

| Layers | Epochs / patience | Target | Epochs run | Train R² | Test R² | Power at \|ω\|=2 |
|---|---|---|---|---|---|---|
| 1 | 100 / 15 (first-run setting) | data | 23 | 0.891 | 0.833 | 0 |
| 1 | 600 / 60 | data | 161 | 0.941 | **0.905** | 0 |
| 2 | 600 / 60 | data | 86 | 0.701 | 0.563 | 32 % |
| 3 | 600 / 60 | data | 117 | 0.675 | 0.544 | 29 % (+6 % at \|ω\|=3) |
| 1 | 600 / 60 | teacher | 375 | 0.971 | 0.967* | 0 |
| 2 | 600 / 60 | teacher | 164 | 0.705 | 0.592* | 34 % |

\* R² against the teacher function, not the observed water level.

## Findings

1. **The hypothesis was wrong. Frequencies do not limit the 1-layer circuit.** Its spectrum
   already supports R² 0.932, as high as the GP. Adding |ω_j| = 2 gains only 0.009.
2. **Feature interactions matter; higher harmonics barely do.** Going from additive to full
   1-layer frequencies lifts the ceiling from 0.899 to 0.932. The cross terms ω = (±1, ±1, …) are
   real physics. Tidal speeds are integer combinations of a few astronomical frequencies
   (K1 + O1 = M2 exactly), so combinations of phases reproduce the constituents that the five
   inputs leave out. The angle kernel uses all 121 cross terms, which is why it reaches 0.927.
3. **The VQC's shortfall is mostly training budget.** Under the first-run settings (patience 15),
   early stopping ended training at epoch 23 with the best weights from epoch 7. With patience 60
   the same circuit reaches 0.905, which beats ridge, but it is still short of the 0.925 ceiling
   for n = 1000.
4. **The circuit's limited reach costs a little more.** Even on a noise-free target inside its
   spectrum, the 32-parameter circuit recovers 97 % of the variance. The 121-frequency Fourier
   model has 243 free coefficients, so the circuit cannot set them all independently.
5. **Deeper circuits are worse, not better.** With 2–3 layers the trained circuits put about
   30 % of their variance at |ω| = 2. The data does not need this; in the teacher test the target
   has none at all. These circuits also underfit the training set (train R² ≈ 0.70). Extra layers
   add frequencies the problem doesn't need and make optimization harder. That explains why the
   tuner always chose 1 layer.
6. Band-limit check: every model spectrum has zero power (< 1e-30) above |ω| = L, as theory
   predicts.

## Caveats

* The probes use one seed and one training subsample. The direction of each effect is clear, but
  the exact numbers will vary.
* We do not split the *data's* variance by frequency order. Tidal phases lie on one time line, so
  many frequency vectors coincide (K1 + O1 = M2) and their individual coefficients are not
  identifiable. The nested ceilings compare spans only, so they do not have this problem.
* **Re-run.** Finding 3 led to a re-run of every VQC benchmark run with a longer budget (batches of 128,
  up to 400 epochs, patience 40). On tides at n = 1000 the mean test R² rose from 0.870 ± 0.026
  to 0.894 ± 0.013, with large gains at small n on wind and photo-z and little change on soccer.
  It is still below ridge (0.899) and the 0.925 ceiling for n = 1000. The final tides fits ran a
  median of 373 of the 400 epochs, so the circuit was probably still improving. The v1 runs are
  kept in `results/archive/vqc_v1/`.
