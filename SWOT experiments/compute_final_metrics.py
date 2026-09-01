"""
Part of the FINAL_PLOTS_AND_METRICS pipeline: the project's headline
accuracy table -- RMSE/bias/variance-of-differences (VOD) for each of the
3 standing models (DRF fixed-window, DRF sliding-window, DUACS) against
real SWOT points over the full July 2025 (31-day) validation window,
broken down by 4 regions and, for bias/VOD only, by 2 aggregation levels.

Pooled bias/VOD (the `<region>_rmse`/`<region>_bias`/`<region>_vod`
columns) are exactly what build_validation_data.py's own summary.csv
already computes via _rmse_bias_variance -- mean/variance of every
individual point's error (model_pred - swot_ssha), pooled flat across
every point in that region over the whole month.

Per-pass bias/VOD (`<region>_per_pass_bias`/`<region>_per_pass_vod`)
group points by (cycle, pass) -- one group per half-orbit SWOT swath --
compute bias (mean error) and VOD (population variance of error, ddof=0,
matching _rmse_bias_variance's np.nanvar) *within* each pass separately,
then average those per-pass values evenly across every pass that has at
least one point in that region -- every pass gets exactly one vote
regardless of how many points it contributed (a pass with 1,611 points
counts the same as one with 457,765). This is standard altimetry
cross-validation practice (errors within one pass are spatially
correlated -- orbit error, tides, mesoscale features -- so pooling
everything flat can be dominated by a handful of noisy passes); it is a
genuinely different number from the pooled version, not a re-derivation
of it. Deliberately no minimum-points-per-pass threshold: a pass
contributing only 1-2 points to a region still gets exactly one equal
vote in that region's per-pass average (a single-point pass's own
per_pass_vod is trivially 0, a known, accepted source of noise for now --
flagged during design, not fixed here since it was explicitly requested
to be left alone).

`<region>_per_pass_bias_weighted`/`<region>_per_pass_vod_weighted` are
the same per-pass grouping, but weighted by each pass's own point count
instead of one equal vote each -- see the law-of-total-variance comment
in _compute_region_metrics. per_pass_vod_weighted is the genuinely new
number here (mathematically guaranteed <= the pooled vod, the "within-
pass" component of it); per_pass_bias_weighted is, by the same identity,
always numerically identical to the plain pooled bias column -- included
because it was asked for, and because matching it is a useful correctness
check on the weighting itself, but it carries no new information over
`<region>_bias`.

Regions are built from the coastal_flag/high_var_flag columns already in
points_validation.parquet (see build_coastal_mask.py's/
build_variance_mask.py's own -1/0/1 conventions -- -1 always means
land/no-data) as 3 mutually exclusive buckets:
  - coastal:            coastal_flag == 1
  - offshore_high_var:  coastal_flag == 0 and high_var_flag == 1
  - offshore_low_var:   coastal_flag == 0 and high_var_flag == 0
A point with coastal_flag == -1 (land/no-data -- rare for a real SWOT
ocean observation, but possible right at the coastline) falls into none
of the 3 regional buckets, but is still included in "overall" (no region
filter at all).

Output is one row per model, with columns = 1 model-name column + 4
regions x 7 metrics (rmse, bias, vod, per_pass_bias, per_pass_vod,
per_pass_bias_weighted, per_pass_vod_weighted) = 29 columns total, saved to
validation_results/FINAL_PLOTS_AND_METRICS/final_metrics.csv.

Usage:
    python compute_final_metrics.py
"""

import math

import pandas as pd

from build_validation_data import VALIDATION_RESULTS_DIR, _rmse_bias_variance

FIXED_DIR_NAME = "val_july25_fixed_realv2"
SLIDING_DIR_NAME = "val_july25_sliding_realv1"
FINAL_PLOTS_DIR = VALIDATION_RESULTS_DIR / "FINAL_PLOTS_AND_METRICS"
OUTPUT_PATH = FINAL_PLOTS_DIR / "final_metrics.csv"

REGIONS = ["overall", "coastal", "offshore_high_var", "offshore_low_var"]

_FIXED_COLUMNS = ["cycle", "pass", "coastal_flag", "high_var_flag", "drf_run", "drf_error", "duacs_run", "duacs_error"]
_SLIDING_COLUMNS = ["cycle", "pass", "coastal_flag", "high_var_flag", "drf_run", "drf_error"]


def _round_sig(x, sig=3):
    """Rounds to `sig` significant figures rather than decimal places --
    same helper as experiment_5.py's own _round_sig, duplicated here
    rather than imported to avoid pulling that script's heavy
    training-only dependencies in just for a small rounding utility (same
    reasoning replot_grid.py/build_experiment_data.py already apply to
    their own duplicated code -- see CLAUDE.md)."""
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def _round_df_sig_figs(df, sig=3):
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].apply(lambda x: _round_sig(x, sig))
    return df


def _region_mask(df, region):
    if region == "overall":
        return pd.Series(True, index=df.index)
    if region == "coastal":
        return df["coastal_flag"] == 1
    if region == "offshore_high_var":
        return (df["coastal_flag"] == 0) & (df["high_var_flag"] == 1)
    if region == "offshore_low_var":
        return (df["coastal_flag"] == 0) & (df["high_var_flag"] == 0)
    raise ValueError(f"Unknown region {region!r}")


def _compute_region_metrics(df, error_col, region):
    # dropna up front (rather than relying on groupby/.mean()/.var() to
    # skip NaN internally, which they do) so every group's own `counts`
    # below is guaranteed > 0 -- otherwise a pass with zero non-null
    # errors (e.g. a masked DUACS pixel) would still appear as a group
    # with mean/var == NaN and count == 0, and NaN * 0 in the weighted
    # sums below would poison the whole region's weighted average instead
    # of just being correctly excluded.
    sub = df.loc[_region_mask(df, region), ["cycle", "pass", error_col]].dropna(subset=[error_col])

    rmse, bias, vod = _rmse_bias_variance(sub[error_col].to_numpy())

    per_pass = sub.groupby(["cycle", "pass"])[error_col]
    counts = per_pass.count()
    means = per_pass.mean()
    variances = per_pass.var(ddof=0)

    per_pass_bias = float(means.mean())
    per_pass_vod = float(variances.mean())
    # Weighted by each pass's own point count, matching the law of total
    # variance: pooled_vod == weighted_mean(per_pass_vod) +
    # weighted_var(per_pass_bias), so per_pass_vod_weighted is
    # mathematically guaranteed <= vod (the "within-pass" component of
    # the pooled variance), unlike the plain per_pass_vod above, which is
    # an equal vote per pass regardless of size and has no such guarantee.
    # per_pass_bias_weighted is, by the same identity, always exactly
    # equal to bias (a size-weighted mean of group means is just the
    # pooled mean) -- included anyway since it was asked for, and it's a
    # useful cross-check that this weighting is implemented correctly.
    per_pass_bias_weighted = float((means * counts).sum() / counts.sum())
    per_pass_vod_weighted = float((variances * counts).sum() / counts.sum())

    return {
        f"{region}_rmse": rmse,
        f"{region}_bias": bias,
        f"{region}_vod": vod,
        f"{region}_per_pass_bias": per_pass_bias,
        f"{region}_per_pass_vod": per_pass_vod,
        f"{region}_per_pass_bias_weighted": per_pass_bias_weighted,
        f"{region}_per_pass_vod_weighted": per_pass_vod_weighted,
    }


def _compute_model_row(model_name, df, error_col):
    row = {"model": model_name}
    for region in REGIONS:
        row.update(_compute_region_metrics(df, error_col, region))
    return row


def main():
    fixed_out_dir = VALIDATION_RESULTS_DIR / FIXED_DIR_NAME
    sliding_out_dir = VALIDATION_RESULTS_DIR / SLIDING_DIR_NAME

    fixed_points_df = pd.read_parquet(fixed_out_dir / "points_validation.parquet", columns=_FIXED_COLUMNS)
    fixed_label = fixed_points_df["drf_run"].iloc[0]
    duacs_label = fixed_points_df["duacs_run"].iloc[0]
    sliding_points_df = pd.read_parquet(sliding_out_dir / "points_validation.parquet", columns=_SLIDING_COLUMNS)
    sliding_label = sliding_points_df["drf_run"].iloc[0]

    print(f"Computing metrics for {fixed_label} ({len(fixed_points_df)} points)...")
    fixed_row = _compute_model_row(fixed_label, fixed_points_df, "drf_error")
    print(f"Computing metrics for {sliding_label} ({len(sliding_points_df)} points)...")
    sliding_row = _compute_model_row(sliding_label, sliding_points_df, "drf_error")
    print(f"Computing metrics for {duacs_label} ({len(fixed_points_df)} points)...")
    duacs_row = _compute_model_row(duacs_label, fixed_points_df, "duacs_error")

    results_df = pd.DataFrame([fixed_row, sliding_row, duacs_row])
    results_df = _round_df_sig_figs(results_df)

    FINAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
