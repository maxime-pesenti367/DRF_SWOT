"""
Drive-relative path resolution, split out of copernicus_pipeline.py so
scripts that only need path resolution (experiment_5.py, replot_grid.py)
don't have to import copernicus_pipeline.py's heavy fetch-only dependencies
(xarray, copernicusmarine, and everything those pull in -- pyarrow,
botocore, dask, zarr, ...) just to call get_drive_base_path(). Deliberately
stdlib-only imports.
"""

import os
import platform
from pathlib import Path


# Returns google drive base path and temp path for temporary zarr storage
def get_drive_base_path(local_drive_letter="G"):

    try:
        # On local Windows, this import will fail and trigger the except block
        import google.colab
        from google.colab import drive

        print("Running in Google Colab. Mounting Drive...")
        drive.mount('/content/drive')
        return Path('/content/drive/MyDrive/SWOT Project/data'), Path('/content/temp_zarr')
    except ImportError:
        if platform.system() == "Windows":
            # Google Drive for Desktop mounts the user's Drive as a drive
            # letter (e.g. G:) on Windows.
            print("Running Locally (Windows).")
            return Path(f"{local_drive_letter}:/My Drive/SWOT Project/data"), Path('./data/temp')
        else:
            # No Windows-style Drive mount exists here (e.g. a remote Linux
            # GPU cluster) -- fall back to the folder that gets synced via
            # `rclone copy gdrive:"SWOT Project/data/..." ~/DRF_SWOT/"SWOT
            # experiments"/gdrive_data/...`, mirroring the same "data" root
            # as the Windows Drive mount. Override with the SWOT_DATA_DIR
            # env var to point at a different location.
            default_local_base = Path.home() / "DRF_SWOT" / "SWOT experiments" / "gdrive_data"
            local_base = os.environ.get("SWOT_DATA_DIR", str(default_local_base))
            print(f"Running Locally (non-Windows, no Drive mount assumed). Using '{local_base}'.")
            return Path(local_base), Path('./data/temp')
