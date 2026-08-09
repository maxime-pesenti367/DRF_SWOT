# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository structure

This repo contains two related but distinct components for a UCL Masters project on interpolating SWOT (Surface Water and Ocean Topography) satellite altimetry data:

```
DRF_SWOT_project/
├── DeepRandomFeatures/     # Modified fork of external code (totony4real/DeepRandomFeatures)
└── SWOT experiments/       # Original work: data pipelines, notebooks, and experiment scripts
```

- `DeepRandomFeatures/` is a lightly-modified fork of a PhD student's research repo implementing the DRF model. Treat changes here conservatively — prefer minimal, targeted edits over refactors, since this needs to stay comparable to the upstream implementation.
- `SWOT experiments/` (note the space in the directory name — always quote it in shell commands) is where the actual masters project work happens: downloading/processing Copernicus Marine and SWOT satellite data, and running DRF experiments on it.

There is no top-level build system, test suite, or CI tying the two together — each experiment is invoked directly as a script or notebook with a YAML config.

## Setup and running code

```bash
# From DeepRandomFeatures/
python3 -m venv venv
source venv/bin/activate          # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
pip install -e ./                 # installs the `DRF` package (src/DRF) in editable mode
```

`SWOT experiments/` has no separate requirements file — it imports the `DRF` package installed above plus `copernicusmarine`, `xarray`, `cartopy`, `pandas`, etc. Install those manually as needed.

Running an experiment (from `DeepRandomFeatures/`):
```bash
python examples/experiment_1.py --config configs/example_config_exp1.yaml
python examples/experiment_2.py --config configs/example_config_exp2.yaml
python examples/experiment_3.py --config configs/example_config_exp3.yaml
```

Running the SWOT-specific experiment (from `SWOT experiments/`, config path is resolved relative to the script via `Path(__file__)`, not the cwd):
```bash
python experiment_4.py
```

There are no automated tests, linter, or formatter configured in this repository.

### Data

All data lives outside version control:
- `DeepRandomFeatures/gdrive_data/` (exp1–exp3 datasets, downloaded from a Google Drive link listed in `DeepRandomFeatures/README.md`) is gitignored.
- `SWOT experiments/data` is gitignored.
- Several scripts/configs hard-code absolute local paths (e.g. `G:/My Drive/SWOT Project/data/...`, `experiment_4.py`, `configs/config_exp4.yaml`) — these are per-machine paths from the original author and will need updating to run locally.
- `SWOT experiments/how to download swot data.md` documents using the PO.DAAC `podaac-data-downloader` CLI (netrc-based auth against `urs.earthdata.nasa.gov`) as one data source; `copernicus_pipeline.py` covers pulling data via the `copernicusmarine` API as another.

## DeepRandomFeatures architecture

The package lives at `DeepRandomFeatures/src/DRF/` (installed as `DRF`). Core idea: approximate stationary Gaussian Process kernels via random feature expansions, used as NN layers, then combine spatial + temporal feature branches to predict a scalar field (e.g. sea surface height anomaly) with uncertainty.

- **`layers.py`** — `RFFLayer` (abstract base): samples a fixed (non-trainable) hidden linear layer from a kernel's spectral density, applies a cosine activation, then a trainable linear output layer.
  - `SquaredExponentialRFFLayer` — samples from a Normal distribution (RBF kernel).
  - `MaternRFFLayer` — samples from a Student-t distribution (Matérn kernel), parameterized by `nu`.
  - `RandomPhaseFeatureMap` / `MaternRandomPhaseS2RFFLayer` — a separate, non-`RFFLayer` implementation for **spherical** Matérn GPs (features on S²), used for global/spherical spatial data (longitude/latitude, optionally converted to Cartesian). Built on `geometric_kernels` (spherical harmonics + Gegenbauer polynomials).
  - `get_layer(name, ...)` in `__init__.py` is the factory for planar layers (`"SquaredExponential"`, `"Matern"`).

- **`models.py`** — `SpatiotemporalModelBase`: processes spatial and temporal inputs through separate stacks of layers, then combines them (`combine_type`: `concat`/`product`/`sum`) before a final linear output layer.
  - `DeepSpatiotemporalGPNN` — planar model; stacks spatial RFF layers with skip connections (concatenating the original spatial input at each layer).
  - `DeepMaternRandomPhaseS2RFFNN` — spherical model; first spatial layer is `MaternRandomPhaseS2RFFLayer`, subsequent layers use `SumFeatures` (combines a planar Matérn branch with the S² random-phase branch). This is the model used by both `experiment_3.py` and `experiment_4.py` (i.e. the actual SWOT work).
  - `initialize_model(model_name, ...)` is the factory (`"DeepSpatiotemporalGPNN"` or `"DeepMaternRandomPhaseS2RFFNN"`).

- **Uncertainty quantification / training loops** — two parallel implementations exist for different model types:
  - `uq_methods.py` → `DeepEnsemble`: trains an ensemble of `DeepSpatiotemporalGPNN`-style models in parallel via `joblib.Parallel`, using a MAP loss (MSE + L2 weight regularization). Used together with `BayesianOptimiser.py`.
  - `spherical_uq_methods.py` → `SphericalBayesianOptimizer` + `train_model_process`: trains an ensemble of spherical models via `torch.multiprocessing` (spawn), using Huber loss plus a spatial functional-regularisation penalty (`utils.functional_regularisation_S2_batched`) evaluated over an S² grid (`d_phi`/`d_theta` spacing). This is the path used for the SWOT/spherical experiments.
  - Both expose a Bayesian hyperparameter optimizer (via `botorch`/`gpytorch`'s `SingleTaskGP` + Expected Improvement) that searches over lengthscales/amplitudes, treating the ensemble's validation loss as the black-box objective.

- **`data_utils.py`** — dataset-specific loaders: `get_mss_data` (exp1, unzips + normalizes), `prepare_tensor_datasets_ABC` (exp2-style pickle data), `get_spherical_data` (exp3 CSV data — converts lon/lat to radians, splits 70/15/15 train/val/val2, loads a separate test-grid `.pt` file). `experiment_4.py` bypasses this module and loads pre-built tensors directly from a `.pt` file with `torch.load`.

- Config-driven: every experiment is described by a YAML file (`configs/*.yaml`) with `data`, `device`, `model` (name + `kwargs`), `bayesian_optimization` (search bounds + iteration counts), `uq_method`, `training`, and `results` (output file paths) sections. Scripts read these positionally by key — no schema validation, so keys must match exactly what the script expects.

## SWOT experiments architecture

`SWOT experiments/` is the applied side: fetching real satellite altimetry data and feeding it through the DRF model above.

- **`copernicus_pipeline.py`** — downloads/updates per-satellite Zarr stores from the Copernicus Marine Service (`copernicusmarine` package) onto Google Drive, keyed by a `SATELLITE_DATASET_MAP` (satellite name + frequency → Copernicus product ID). `get_drive_base_path()` detects Colab vs. local Windows and resolves the Drive mount path accordingly (`local_drive_letter` param controls which Windows drive letter Google Drive is mounted on — default `"G"`). `copernicus_pipeline_old.py` is a superseded variant (`open_dataset` instead of `.subset`) kept for reference.
- **`display_tracks.py`** — plotting/diagnostic utilities for visualizing satellite ground-track density (e.g. `get_spatial_density` computes an equal-area lat/lon histogram in km² for heatmaps).
- **`experiment_4.py`** — the project-specific experiment script; loads pre-processed tensors (spatial/temporal train+test splits, plus saved normalization stats) from a hard-coded `.pt` path, builds a `DeepMaternRandomPhaseS2RFFNN` via `SphericalBayesianOptimizer`, then plots mean predictions and variance on a `cartopy.PlateCarree` projection using **scatter** (not `imshow`, since SWOT ground tracks are sparse/irregular, unlike the gridded exp3 data). Contains duplicate/leftover plotting blocks (an old `imshow`-based version follows the scatter version) — check which block is actually intended when editing.
- **Notebooks** (`all_satellites_test.ipynb`, `drive_test.ipynb`, `swot_data_test.ipynb`, `swot_expert_data_test.ipynb`) — exploratory/interactive counterparts to the pipeline scripts; not part of any automated flow.
- `configs/config_exp4.yaml` follows the same schema as the DRF `configs/example_config_exp3.yaml` (same model/training/bayesian_optimization keys) plus a `data.tensor_data_path` pointing at the pre-built `.pt` tensor file instead of a raw CSV.
