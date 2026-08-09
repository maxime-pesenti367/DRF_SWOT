
# THIS IS WITH .open_dataset INSTEAD OF .subset
# Downloads and updates per-satellite Zarr stores on Google Drive
# based on a provided DatasetSpec/Config
def fetch_and_store_satellites(config: dict, force_redownload: bool = False):
    drive_base, temp_dir = get_drive_base_path()
    drive_base.mkdir(parents=True, exist_ok=True)

    satellites = config["satellites"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    variables = config["variables"]

    for sat_dict in satellites:
        sat_name = sat_dict["name"]
        freq = sat_dict.get("freq", "1hz") # Defaults to 1hz if not specified

        if sat_name not in SATELLITE_DATASET_MAP:
            print(f"Skipping unknown satellite key: {sat_name}")
            continue

        dataset_id = SATELLITE_DATASET_MAP[sat_name][freq]
        zarr_name = f"{sat_name}_{freq}_{start_date[:4]}_{end_date[:4]}.zarr"
        final_drive_path = drive_base / sat_name / zarr_name

        # Check if already downloaded
        if final_drive_path.exists() and not force_redownload:
            print(f"Skipping {sat_name}: Dataset already exists at {final_drive_path}")
            continue

        print(f"\n--- Processing {sat_name} ({start_date} to {end_date}) ---")
        
        # 1. Connect & Stream via Copernicus
        ds = copernicusmarine.open_dataset(
            dataset_id=dataset_id,
            variables=variables
        )

        # 2. Slice in memory
        ds_sliced = ds.sel(time=slice(start_date, end_date))

        # 3. Write to local SSD temp folder
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
        print(f"Writing temporary Zarr to local SSD...")
        ds_sliced.to_zarr(temp_dir, mode='w')

        # 4. Copy completed Zarr folder to Google Drive
        final_drive_path.parent.mkdir(parents=True, exist_ok=True)
        if final_drive_path.exists():
            shutil.rmtree(final_drive_path)

        print(f"Transferring {zarr_name} to Google Drive...")
        shutil.copytree(temp_dir, final_drive_path)

        # Clean up local temp
        shutil.rmtree(temp_dir)
        print(f"Successfully saved {sat_name} to Drive!")


# THIS IS USING in sys.modules INSTEAD OF ERROR CATCHING TO DETECT VM OR LOCAL
if 'google.colab' in sys.modules:
    # Running in cloud
    print("Running in Google Colab. Mounting Drive...")
    from google.colab import drive
    drive.mount('/content/drive')

    print("Please set paths")
    #BASE_DRIVE_PATH = Path('/content/drive/MyDrive/drf_data')
    #LOCAL_TEMP_PATH = Path('/content/temp_zarr')

else:
    # Running locally
    print("Running locally on Windows")

    BASE_DRIVE_PATH = Path('G:/My Drive/SWOT Project/data')
    LOCAL_TEMP_PATH = Path('./data/temp')

print(f"Base path set to: {BASE_DRIVE_PATH}")