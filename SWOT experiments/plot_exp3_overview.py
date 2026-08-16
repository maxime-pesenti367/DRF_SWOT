"""
Temporary script: reuses exp5's 4-panel display_1D_tracks() plot on exp3's
CSV data, for a like-for-like look at exp3 vs exp5 datasets.

exp3's data is a flat CSV (ssha_20_ku, lat_20_ku, lon_20_ku, time), not the
xarray Dataset display_1D_tracks() expects (sla_filtered/latitude/longitude/
time on a shared "time" dim, matching combine_for_drf's xarray output) -- so
this converts the CSV into that shape in memory before plotting. No files
are written to disk during conversion.

Note: exp3's "time" column is only day-granularity (days since 2000-01-01,
integer) -- see conversation history. The time-series panel will show
values stacked on just a handful of distinct x-positions as a result; this
is a real property of the data, not a bug in this script.

Usage:
    python plot_exp3_overview.py
    python plot_exp3_overview.py --csv "..\DeepRandomFeatures\gdrive_data\exp3.csv" --max-points 500000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from display_tracks import display_1D_tracks

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "DeepRandomFeatures" / "gdrive_data" / "exp3.csv"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "DeepRandomFeatures" / "results" / "exp3" / "tracks_overview.png"
EPOCH = pd.Timestamp("2000-01-01")


def main():
    parser = argparse.ArgumentParser(description="Plot exp3's CSV data with exp5's 4-panel tracks overview")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Path to exp3.csv")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Where to save the overview PNG")
    parser.add_argument("--max-points", type=int, default=300_000, help="Random subsample size for plotting (0 = use all rows)")
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows.")

    if args.max_points and len(df) > args.max_points:
        df = df.sample(n=args.max_points, random_state=0).reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows for plotting.")

    time = EPOCH + pd.to_timedelta(df["time"], unit="D")

    ds = xr.Dataset(
        {
            "sla_filtered": ("time", df["ssha_20_ku"].to_numpy(dtype=np.float32)),
            "latitude": ("time", df["lat_20_ku"].to_numpy(dtype=np.float32)),
            "longitude": ("time", df["lon_20_ku"].to_numpy(dtype=np.float32)),
        },
        coords={"time": time.to_numpy()},
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    display_1D_tracks(ds, "exp3", save_path=out_path)


if __name__ == "__main__":
    main()
