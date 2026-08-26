"""
Tiny, dependency-light helper (pandas + stdlib only -- no torch, no
copernicusmarine) shared between build_sliding_window_data.py (fetch venv)
and run_sliding_window.py (main/train venv, often a different machine
entirely). Split out for the same reason drive_paths.py was split out of
copernicus_pipeline.py: run_sliding_window.py needs to compute the same
target-day sequence build_sliding_window_data.py used, without transitively
importing copernicus_pipeline.py's fetch-only dependencies (importing a
module runs its own imports too, and copernicusmarine isn't installed in
the main/train venv).
"""

import pandas as pd


def generate_target_dates(master_config):
    """Returns the list of target-day date strings ("YYYY-MM-DD") a
    sliding-window master config (data- or exp-side) describes, from its
    start_date/end_date/step_days. This is the single source of truth for
    "which days does this sliding-window model cover" -- both the fetch
    side (which additionally needs sliding_day_range to turn a target day
    into a fetch window) and the train side (which just needs the dates
    themselves) derive their day lists from this."""
    target_days = pd.date_range(
        master_config["start_date"], master_config["end_date"], freq=f"{master_config['step_days']}D",
    )
    return [d.strftime("%Y-%m-%d") for d in target_days]
