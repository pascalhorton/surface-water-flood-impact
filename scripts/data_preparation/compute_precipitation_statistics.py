"""
Extracts the precipitation time series from the raw data and saves it as a netCDF files.
"""
import xarray as xr
from pathlib import Path

from swafi.config import Config
from swafi.precip_combiprecip import CombiPrecip

config = Config()

year_start = 2005
year_end = 2022


def main():
    # Load CombiPrecip files
    precip = CombiPrecip(year_start=year_start, year_end=year_end)
    precip.prepare_data(config.get('DIR_PRECIP'))

    # Compute statistics on the original data (full domain)
    mean, std = precip.compute_mean_and_std_per_pixel()
    q99 = precip.compute_quantile_per_pixel(0.99)

    # Compute the statistics on the log-transformed data
    precip.log_transform()
    mean_log, std_log = precip.compute_mean_and_std_per_pixel()
    q99_log = precip.compute_quantile_per_pixel(0.99)

    # Save the statistics to a netCDF file
    ds_stats = xr.Dataset(
        {
            'mean': (('y', 'x'), mean),
            'std': (('y', 'x'), std),
            'q99': (('y', 'x'), q99),
            'mean_log': (('y', 'x'), mean_log),
            'std_log': (('y', 'x'), std_log),
            'q99_log': (('y', 'x'), q99_log),
        },
        coords={
            'x': precip.data.x,
            'y': precip.data.y,
        }
    )
    output_dir = Path(config.get('OUTPUT_DIR'))
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / f'cpc_statistics_{year_start}-{year_end}.nc'
    print(f'Saving statistics to {stats_path}...')
    ds_stats.to_netcdf(stats_path)

if __name__ == '__main__':
    main()
