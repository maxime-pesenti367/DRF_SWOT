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

**Windows GPU note:** `pip install torch` on Windows defaults to a CPU-only
build, even on a machine with a working NVIDIA GPU/driver (unlike Linux,
where the default plain install pulls a CUDA-enabled build automatically).
If `torch.cuda.is_available()` returns `False` despite `nvidia-smi` showing
a working GPU, reinstall torch from the CUDA wheel index, e.g.:

    pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

Check your GPU driver's CUDA version (`nvidia-smi`) to pick a matching or
older `cuXXX` index — newer drivers are backward compatible with older
CUDA runtimes.

```bash
# From the repo root (DRF_SWOT_project/) -- NOT from DeepRandomFeatures/.
# There is a single venv for the whole repo; requirements.txt at the repo
# root covers both DeepRandomFeatures and SWOT experiments dependencies
# (DeepRandomFeatures/requirements.txt no longer exists -- it was
# consolidated here at some point, so ignore any older instructions that
# reference it).
python3 -m venv venv
source venv/bin/activate          # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
pip install -e ./DeepRandomFeatures    # installs the `DRF` package (src/DRF) in editable mode
```

`requirements.txt` pins `numpy==1.26.4` deliberately — `geomstats` (a `geometric_kernels` dependency) breaks on numpy 2.x (`ImportError: cannot import name 'trapz' from 'numpy'`, removed in numpy 2.0). This pin can silently drift: `copernicusmarine` has been observed to pull numpy forward past the pin on a later `pip install`/upgrade without anyone re-running `pip install -r requirements.txt` afterward. If `from DRF.models import ...` or similar starts failing with a numpy/geomstats import error, re-run `pip install -r requirements.txt` in the venv before debugging further — check `pip show numpy` first to confirm drift.

### Fetching data needs a second, separate venv

`copernicusmarine>=2.4.1` requires `numpy>=2.1.0` (confirmed via `pip install --dry-run`: *"copernicusmarine 2.4.1 depends on numpy>=2.1.0"*) — a hard conflict with the `numpy==1.26.4` pin above. You cannot have a working `copernicusmarine` and a working `geomstats`/DRF in the same venv. Versions of `copernicusmarine` old enough to coexist with `numpy==1.26.4` (e.g. 2.3.0) predate that package's support for downloading sparse/SQLite-backed datasets (several Copernicus altimetry products, including some at 1Hz) as NetCDF — `subset(file_format="netcdf")` fails on them with `WrongFormatRequested: Requested format 'netcdf' is not supported yet`.

So: **`build_experiment_data.py` (and anything else importing `copernicus_pipeline.py`) needs to run from a different venv than the one used for `experiment_5.py`/DRF training.** Set it up once, outside this repo (so it's never confused with the training `venv/` or picked up by git):

```bash
python3 -m venv ../swot-fetch-env         # any location outside the repo works; an existing
                                            # env named swot_env may already exist from earlier setup
../swot-fetch-env/Scripts/activate         # Windows; `source .../bin/activate` elsewhere
pip install -r requirements-fetch.txt
```

Then, from `SWOT experiments/`, with that venv activated instead of the main one:
```bash
python build_experiment_data.py --config configs/data/all_sats_1_day.yaml
```

`requirements-fetch.txt` intentionally has no numpy pin — letting `copernicusmarine` pull in whatever numpy it needs is the entire point of keeping this venv separate. `experiment_5.py`/`replot_grid.py` still run from the main `venv` as before — they only need `copernicus_pipeline.get_drive_base_path()` for path resolution, not the fetching functions themselves, so they're unaffected by which `copernicusmarine` version (if any) is installed there.

Running an experiment (from `DeepRandomFeatures/`):
```bash
python examples/experiment_1.py --config configs/example_config_exp1.yaml
python examples/experiment_2.py --config configs/example_config_exp2.yaml
python examples/experiment_3.py --config configs/example_config_exp3.yaml
```

Running the SWOT-specific experiments (from `SWOT experiments/`):
```bash
python experiment_4.py                                                              # legacy, hard-coded paths — see below
python build_experiment_data.py --config configs/data/all_sats_1_day.yaml            # step 1: fetch + split + tensorize a dataset -- separate fetch venv, see below
python experiment_5.py --config configs/exp5/exp_all_sats_1_day_random_shallow.yaml  # step 2: train/eval/checkpoint against it -- main venv
python replot_grid.py --results-dir results/exp_all_sats_1_day_random_shallow        # optional: regenerate grid plots from saved checkpoints, no retraining -- main venv
```
`experiment_4.py`'s config path is resolved relative to the script via `Path(__file__)`, not the cwd. `experiment_5.py`/`build_experiment_data.py` resolve tensor paths dynamically via `get_drive_base_path()` (see Data section below), so they work unmodified across machines.

There are no automated tests, linter, or formatter configured in this repository.

### Data

All data lives outside version control:
- `DeepRandomFeatures/gdrive_data/` (exp1–exp3 datasets, downloaded from a Google Drive link listed in `DeepRandomFeatures/README.md`) is gitignored.
- `SWOT experiments/data` is gitignored.
- `experiment_4.py`/`configs/config_exp4.yaml` hard-code absolute local paths (e.g. `G:/My Drive/SWOT Project/data/...`) — per-machine paths from the original author, left as-is since exp4 is legacy (see below); these will need updating to run exp4 locally.
- `SWOT experiments/how to download swot data.md` documents using the PO.DAAC `podaac-data-downloader` CLI (netrc-based auth against `urs.earthdata.nasa.gov`) as one data source; `copernicus_pipeline.py` covers pulling data via the `copernicusmarine` API as another.
- **`get_drive_base_path(local_drive_letter="G")`** in `copernicus_pipeline.py` is the single source of truth for where satellite/tensor data lives, and is platform-aware (three branches): Colab (mounts Drive at a fixed path), Windows (assumes Google Drive for Desktop is mounted as a drive letter, default `G:`), and everything else (e.g. a remote Linux GPU cluster with no Drive mount — falls back to `SWOT_DATA_DIR` env var, or `./data`/`~/DRF_SWOT/SWOT experiments/gdrive_data` if unset). Exp5's pipeline (`build_experiment_data.py` writes, `experiment_5.py`/`replot_grid.py` read) always resolves paths through this function rather than hard-coding them, specifically so the same config works on any machine — sync data between machines (e.g. Windows Drive ↔ Linux cluster) with `rclone`/`scp` into whatever `get_drive_base_path()` resolves to on the target machine.

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

**exp5 is the current, actively-developed pipeline and the intended long-term replacement for exp3/exp4** — config-driven end-to-end, with a real train/val/test split, correct normalization for this project's data (no z-scoring bug — see below), model weight checkpointing (nothing before exp5 in this codebase ever saved trained weights), and real test-set accuracy metrics (RMSE/NLPD/CRPS). exp3/exp4 are kept for reference/comparison, not further developed — **treat `experiment_3.py` (in `DeepRandomFeatures/examples/`) and `experiment_4.py` as protected/do-not-touch** unless explicitly asked, since ongoing investigations (see `exp3` vs `exp5` comparison docs, referenced by whoever's continuing this work) depend on them staying as-is for a stable baseline.

- **`copernicus_pipeline.py`** — shared low-level pipeline used by both exp4 and exp5. `fetch_and_store_satellites`/`load_experiment_dataset` download/cache per-satellite Zarr stores from Copernicus Marine (`copernicusmarine` package) keyed by a `SATELLITE_DATASET_MAP` (satellite name + frequency → Copernicus product ID) and cached by satellite+date-range (independent of any "experiment name"). `combine_for_drf` merges multiple satellites' data into one dataframe/xarray dataset. `store_tensors`/`get_drive_base_path` handle Drive-relative tensor storage — see the Data section above for `get_drive_base_path`'s platform-aware behaviour. `process_and_split_dataframe` is **not** used by exp5 — it bakes together a 2-way split with spatial z-score normalization that breaks `spherical_to_cartesian()`'s geometry (a known bug, deliberately not fixed in exp4 where it still exists — exp5's own tensorization was written from scratch to avoid it). `copernicus_pipeline_old.py` is a superseded variant (`open_dataset` instead of `.subset`) kept for reference.
- **`display_tracks.py`** — plotting/diagnostic utilities for visualizing satellite ground-track density (e.g. `get_spatial_density` computes an equal-area lat/lon histogram in km² for heatmaps).
- **`experiment_4.py`** *(legacy, protected — see above)* — loads pre-processed tensors from a hard-coded `.pt` path, builds a `DeepMaternRandomPhaseS2RFFNN` via `SphericalBayesianOptimizer`, plots mean/variance via `scatter` over the sparse real test points, plus a whole-globe grid snapshot (`imshow`) retrained separately since no weights were ever saved. Has the known spatial z-score normalization bug mentioned above.
- **`data_splits.py`** — pure split functions used by `build_experiment_data.py`, no I/O: `random_split`, `temporal_block_split` (train on data before a cutoff date, test after), `spatial_block_split` (train outside a bounding box, test inside it — e.g. holding out a specific ocean region). `SPLIT_METHODS` dict maps config `method` strings to these functions. Block splits test genuine spatial/temporal extrapolation rather than the interpolation-friendly leakage a random point-wise split allows.
- **`build_experiment_data.py`** — step 1 of the exp5 pipeline. Takes a **data config** (`configs/data/*.yaml`: satellites, date range, variables, list of named split strategies), orchestrates the existing fetch/load/combine pipeline from `copernicus_pipeline.py`, applies each configured split via `data_splits.py`, then normalizes/tensorizes itself (spatial: degrees→radians only, no z-scoring; temporal: Unix seconds z-scored using the *train* split's own mean/std; target: left raw). Saves one `.pt` file per split to `<drive_base>/pytorch tensors/<data-config-name>/<split-name>.pt` via `store_tensors`. Logs split sizes and warns (not errors) on empty splits.
- **`model_io.py`** — `save_checkpoint`/`load_checkpoint`: bundles a trained model's `state_dict` + architecture config + winning hyperparameters + normalization stats + seed into one `.pt` file, enough to fully reconstruct and reuse the model later without retraining.
- **`experiment_5.py`** — step 2 of the exp5 pipeline; the canonical training/eval script. Loads a **model config** (`configs/exp5/*.yaml`: which data config + split to use via `data.experiment_name`/`data.split_name`, model architecture, BO search settings, training settings, optionally `seed` and `training.max_parallel_models`), runs `SphericalBayesianOptimizer` (from `DRF.spherical_uq_methods`, shared with exp3/exp4) to search hyperparameters against a real held-out val split, then retrains a fresh final ensemble with the winning hyperparameters (`train_final_model`, a deliberate small duplication of `train_model_process`'s loop — needed because that shared function only ever returns predictions, never the model object itself). Scores the final ensemble against the real held-out test set (RMSE/NLPD/CRPS — genuinely new capability, exp3/exp4 never did this), saves per-model checkpoints, prediction tensors, and a `results.csv`, then plots a whole-globe grid snapshot (`imshow`, matching exp3/exp4's style) by forward-passing the already-trained final ensemble over a synthetic dense grid — no separate retrain needed, unlike exp4, since the weights are already saved. Results land in `results/<model-config-name>/`, name auto-derived from the config filename (never hand-typed).
- **`replot_grid.py`** — standalone: regenerates a saved exp5 run's whole-globe grid PNGs from its checkpoints alone (`python replot_grid.py --results-dir results/<config-name>`), with zero retraining or BO search. Useful for updating plots for older runs, or after a plotting-code change, without repeating the expensive part.
- **Config naming convention**: data configs live in `configs/data/<dataset-name>.yaml` (e.g. `all_sats_1_day.yaml`); model/experiment configs live in `configs/exp5/exp_<dataset-name>_<split>_<depth>.yaml` (e.g. `exp_all_sats_1_day_random_shallow.yaml`, where `shallow` signals a small BO search budget — `initial_samples`/`n_iterations` — as opposed to a future, more thorough `deep`/`full` variant). The two are linked only by the model config's `data.experiment_name` matching the data config's internal `name:` field — not by filename.
- **Notebooks** (`all_satellites_test.ipynb`, `drive_test.ipynb`, `swot_data_test.ipynb`, `swot_expert_data_test.ipynb`) — exploratory/interactive counterparts to the pipeline scripts; not part of any automated flow.
- `configs/config_exp4.yaml` follows the same schema as the DRF `configs/example_config_exp3.yaml` (same model/training/bayesian_optimization keys) plus a `data.tensor_data_path` pointing at the pre-built `.pt` tensor file instead of a raw CSV.

### GPU memory with `SphericalBayesianOptimizer`

Each Bayesian-optimization candidate trains `num_models` ensemble members concurrently via `torch.multiprocessing.Pool` — GPU memory scales with concurrent worker count, not with dataset size (training is batched; a bigger dataset just means more batches, not more peak memory). If you hit a CUDA OOM, check `bayesian_optimization.initial_samples + n_iterations` (total search rounds) before assuming it's the dataset or model size — more rounds means the pool gets torn down and rebuilt more times, and previously an incomplete teardown (`Pool` used as a context manager calls `terminate()`, not graceful `close()`+`join()`) could let memory creep across rounds; this is now fixed in `spherical_uq_methods.py`. `SphericalBayesianOptimizer` also accepts `max_parallel_models` (defaults to `num_models`, i.e. unchanged behaviour) to cap how many ensemble members train at once, trading wall-clock time for lower peak GPU memory — exposed as `training.max_parallel_models` in exp5 configs.
