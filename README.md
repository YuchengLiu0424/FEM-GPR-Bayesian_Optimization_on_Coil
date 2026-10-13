# FEM–GP Coil Optimization

This repository contains reproducibility artifacts for a physics-constrained electromagnetic coil–core optimization study. The workflow connects Python to a two-dimensional axisymmetric COMSOL model through MPh, fits Gaussian-process (GP) surrogates, and compares expected-improvement Bayesian optimization with several derivative-free reference optimizers under explicit finite-element-method (FEM) evaluation budgets.

## Repository contents

- `code/data/`: sampling, FEM automation, surrogate modeling, optimization, validation, sensitivity, and claim-audit scripts with their reproducibility outputs and figures.
- `surrogate_simulation/`: the source COMSOL `.mph` model.
- `resources/`: project-owned supporting resources, when present.

The unsubmitted manuscript source and compiled draft, personal notes, local editor files, Python caches, and third-party literature PDFs are intentionally excluded from version control.

## Reproducibility status

- Current FEM table: 250 validated designs.
- Quantitative claim audit: 328/328 checks passing.
- Source-model SHA-256: `1c261222bc6281419b1b1d2b24e510c30acc6dbff83d4ceff3f10949d66cfcd3`.
- Fresh-session checks confirm a zero ROI field at zero applied current and do not save or modify the source COMSOL model.

The audit entry point is `code/data/py18_manuscript_claim_audit.py`; its summary is stored in `code/data/py18_output/summary.json`.

## Software

The archived workflow was run with COMSOL Multiphysics 6.3, Python 3.14.6, MPh 1.3.1, NumPy 2.4.3, SciPy 1.17.1, and scikit-learn 1.8.0. A licensed local COMSOL installation is required for scripts that perform new FEM solves.

## License

No reuse license has been assigned yet. All rights are reserved until separate licenses for the code, data/model artifacts, and figures are selected.
