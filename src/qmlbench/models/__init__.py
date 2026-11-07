"""Model zoo. Every model follows the scikit-learn API (fit / predict), so the benchmark
runner treats quantum and classical models identically.

Implementation status
---------------------
classical.py      : ready (thin wrappers around scikit-learn)
quantum_vqc.py    : ready -- data re-uploading regressor (PennyLane + PyTorch)
quantum_kernel.py : ready -- fidelity kernel (angle / IQP feature maps) + kernel ridge
domain.py         : ready -- UTide, NOAA, power curve, logistic xG, StatsBomb xG, kNN photo-z
"""
