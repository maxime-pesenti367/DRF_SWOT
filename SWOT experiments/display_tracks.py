import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr


# Takes in ds, returns spatial_density as well as mesh grid, used for drawing heatmap of track density
def get_spatial_density(ds, grid_res_deg=1.0, heatmap_white_nan=False):

    R = 6371.0  
    
    # 1. Convert to DataFrame safely, whether it's a DataArray or Dataset
    if isinstance(ds, xr.DataArray):
        # We give the single array a dummy name like 'value' so Pandas can structure it
        df = ds.to_dataframe(name='value').reset_index()
    else:
        df = ds.to_dataframe().reset_index()

    # 2. Drop rows where latitude or longitude are missing
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # 3. Extract the clean arrays
    lats = df['latitude'].values
    lons = df['longitude'].values

    lons = np.where(lons > 180, lons - 360, lons)
    lon_edges = np.arange(-180, 180 + grid_res_deg, grid_res_deg)
    lat_edges = np.arange(-90, 90 + grid_res_deg, grid_res_deg)

    counts, _, _ = np.histogram2d(lons, lats, bins=[lon_edges, lat_edges])
    counts = counts.T  

    dlon_rad = np.radians(grid_res_deg)
    lat_bins_rad = np.radians(lat_edges)
    sin_diffs = np.sin(lat_bins_rad[1:]) - np.sin(lat_bins_rad[:-1])
    cell_areas_km2 = (R**2) * dlon_rad * sin_diffs

    area_matrix_km2 = np.tile(cell_areas_km2[:, np.newaxis], (1, len(lon_edges) - 1))

    density = (counts / area_matrix_km2) * 1000.0
    
    if heatmap_white_nan:
        density = np.ma.masked_where(counts == 0, density)
    
    # Gridmesh for pcolormesh
    lon_mesh, lat_mesh = np.meshgrid(lon_edges, lat_edges)

    return density, lon_mesh, lat_mesh



# Display SSHA tracks from 1D altimeter
# Graphs included: Spatial plot, time series, distribution
# Note: ssha must be a xarray.DataArray, with coordinates: time, longitude, latitude. This can be done using assign_coords
def display_1D_tracks(ssha, experiment_name, dot_size=5, vmax=0.2, outlier_threshold=5, include_nan=False, heatmap_res_deg=1, heatmap_white_nan=False, save_path=None):

    # Assign coordinates if not done already
    ssha = ssha['sla_filtered'].assign_coords(
        time=ssha['time'], 
        latitude=ssha['latitude'], 
        longitude=ssha['longitude']
    )

    # Load ssha to load in real values
    ssha.load()

    # Configs
    vmin = -vmax
    cmap = 'RdBu_r'
    distribution_range = 1.0

    time = ssha['time']
    lat = ssha['latitude']
    lon = ssha['longitude']

    # Calculate SSHA basic stats (mean, std, etc.)
    mean_ssha = ssha.mean().item()
    std_ssha = ssha.std().item()
    lower_bound = mean_ssha - outlier_threshold * std_ssha
    upper_bound = mean_ssha + outlier_threshold * std_ssha
    outlier_proportion = np.sum((ssha.values < lower_bound) | (ssha.values > upper_bound)) / len(ssha)
    nan_mask = np.isnan(ssha)

    layout = """
    AB
    CC
    DD
    """

    fig, axes = plt.subplot_mosaic(layout, figsize=(16, 20))

    t_min = pd.to_datetime(time.min().values).strftime('%Y-%m-%d %H:%M:%S')
    t_max = pd.to_datetime(time.max().values).strftime('%Y-%m-%d %H:%M:%S')
    fig.suptitle(f'{experiment_name} SSHA from "{t_min}" to "{t_max}"')

    # A: Time series
    sc2 = axes['A'].scatter(time[~nan_mask], ssha[~nan_mask], c=ssha[~nan_mask], vmin=vmin, vmax=vmax, cmap=cmap, s=dot_size)
    fig.colorbar(sc2, ax=axes['A'], label='SSHA (m)')
    axes['A'].set_title(f'SSHA time series   (Outlier proportion not visible on graph: {outlier_proportion:.2%})')
    axes['A'].set_xlabel('Time')
    axes['A'].set_ylabel('SSHA (m)')
    axes['A'].tick_params(axis='x', rotation=45)
    axes['A'].set_ylim(lower_bound, upper_bound)
    axes['A'].axhline(y=0, color='gray', linestyle=':', alpha=0.3)

    # B: Distribution
    counts, bins = np.histogram(ssha, bins=100, range=(-distribution_range, distribution_range))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    norm = plt.Normalize(vmin, vmax)
    colormap = plt.get_cmap(cmap)
    axes['B'].bar(bin_centers, counts, width=(bins[1] - bins[0]), color=colormap(norm(bin_centers)), edgecolor='black', linewidth=0.3)  
    axes['B'].set_title(f'SSHA Distribution   (mean: {mean_ssha:.2g}, std: {std_ssha:.2g})')
    axes['B'].set_xlabel('SSHA (m)')
    axes['B'].set_ylabel('Pixel Count')
    axes['B'].grid(axis='y', linestyle='--', alpha=0.5)

    # C: Spatial plot
    if include_nan:
        # Plot nans (green)
        axes['C'].scatter(lon[nan_mask], lat[nan_mask], color='green', s=dot_size)
    
    # Plot valid SSHAs
    sc = axes['C'].scatter(lon, lat, c=ssha, vmin=vmin, vmax=vmax, cmap=cmap, s=dot_size)
    fig.colorbar(sc, ax = axes['C'], label='SSHA (m)')
    axes['C'].set_title(f'SSHA spatial plot   (NaN proportion: {np.isnan(ssha).sum() / len(ssha):.2%})')
    axes['C'].set_xlabel('Longitude')
    axes['C'].set_ylabel('Latitude')
    axes['C'].grid(True, linestyle='--', alpha=0.2)


    # D: Heatmap of track density
    density, lon_mesh, lat_mesh = get_spatial_density(ssha, grid_res_deg=heatmap_res_deg, heatmap_white_nan=heatmap_white_nan)

    pcm = axes['D'].pcolormesh(
        lon_mesh, lat_mesh, density, 
        cmap='viridis', 
        shading='flat'
    )
    
    cbar = fig.colorbar(pcm, ax=axes['D'], pad=0.08, shrink=0.7)
    cbar.set_label('Data Density (Points per 1,000 $\mathrm{km}^2$)', fontsize=11)
    
    axes['D'].set_title(f'Heatmap of track density (Grid resolution: {heatmap_res_deg}°)')
    axes['D'].set_xlabel('Longitude')
    axes['D'].set_ylabel('Latitude')
    axes['D'].set_xlim(-180, 180)
    axes['D'].set_ylim(-90, 90)
    axes['D'].grid(True, linestyle='--', alpha=0.3)


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Saved {save_path}")
    else:
        plt.show()



