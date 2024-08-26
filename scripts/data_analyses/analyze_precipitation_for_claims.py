"""
This script analyzes the precipitation data characteristics for each claim.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from swafi.config import Config
from swafi.precip_combiprecip import CombiPrecip
from swafi.damages_mobiliar import DamagesMobiliar
from swafi.damages_gvz import DamagesGvz

config = Config(output_dir='analysis_precip_claims')

DATASET = 'mobiliar'  # 'mobiliar' or 'gvz'
PRECIP_DAYS_BEFORE = 2
PRECIP_DAYS_AFTER = 2

if DATASET == 'mobiliar':
    EXPOSURE_CATEGORIES = ['external']
    CLAIM_CATEGORIES = ['external', 'pluvial']
elif DATASET == 'gvz':
    EXPOSURE_CATEGORIES = ['all_buildings']
    CLAIM_CATEGORIES = ['likely_pluvial']


def main():
    generate_csv()
    generate_plots()


def generate_csv():
    # Load the damage data
    year_start = None
    year_end = None
    if DATASET == 'mobiliar':
        year_start = config.get('YEAR_START_MOBILIAR')
        year_end = config.get('YEAR_END_MOBILIAR')
        damages = DamagesMobiliar(dir_exposure=config.get('DIR_EXPOSURE_MOBILIAR'),
                                  dir_claims=config.get('DIR_CLAIMS_MOBILIAR'),
                                  year_start=config.get('YEAR_START_MOBILIAR'),
                                  year_end=config.get('YEAR_END_MOBILIAR'))
    elif DATASET == 'gvz':
        year_start = config.get('YEAR_START_GVZ')
        year_end = config.get('YEAR_END_GVZ')
        damages = DamagesGvz(dir_exposure=config.get('DIR_EXPOSURE_GVZ'),
                             dir_claims=config.get('DIR_CLAIMS_GVZ'),
                             year_start=config.get('YEAR_START_GVZ'),
                             year_end=config.get('YEAR_END_GVZ'))
    else:
        raise ValueError(f'Dataset {DATASET} not recognized.')

    # Select the categories of interest
    damages.select_categories_type(EXPOSURE_CATEGORIES, CLAIM_CATEGORIES)

    # Compute the precipitation time series for each claim
    claims = damages.claims

    # Extract CIDs with claims
    cids = damages.claims['cid'].unique()

    # Load CombiPrecip files
    precip = CombiPrecip(year_start, year_end)
    precip.load_data(config.get('DIR_PRECIP'))

    # Add columns to the claims dataframe
    claims['precip_max'] = None
    claims['precip_max_q'] = None
    claims['precip_dt'] = None
    claims['precip_06h_max'] = None
    claims['precip_12h_max'] = None
    claims['precip_24h_max'] = None

    for cid in cids:
        print(f'Processing CID {cid}')
        precip_cid_q = precip.compute_quantiles_cid(cid)

        for idx, claim in claims[claims['cid'] == cid].iterrows():
            start = claim['date_claim'] - pd.Timedelta(days=PRECIP_DAYS_BEFORE)
            end = claim['date_claim'] + pd.Timedelta(days=PRECIP_DAYS_AFTER)
            precip_ts = precip.get_time_series(cid, start, end)
            precip_ts = precip_ts.flatten()
            precip_q_ts = precip_cid_q.sel(time=slice(start, end))
            precip_q_ts = precip_q_ts['precip'].to_numpy()

            # Compute the max precipitation
            claims.loc[idx, 'precip_max'] = precip_ts.max()
            claims.loc[idx, 'precip_max_q'] = precip_q_ts.max()

            # Compute centrality of the max precipitation
            if precip_ts.max() > 0:
                idx_max = precip_ts.argmax()
                claims.loc[idx, 'precip_dt'] = idx_max - len(precip_ts) // 2

            # Compute a precipitation sum on rolling windows
            precip_ts_6h = np.convolve(precip_ts, np.ones(6), mode='valid')
            claims.loc[idx, 'precip_06h_max'] = precip_ts_6h.max()
            precip_ts_12h = np.convolve(precip_ts, np.ones(12), mode='valid')
            claims.loc[idx, 'precip_12h_max'] = precip_ts_12h.max()
            precip_ts_24h = np.convolve(precip_ts, np.ones(24), mode='valid')
            claims.loc[idx, 'precip_24h_max'] = precip_ts_24h.max()

    # Save the claims dataframe
    filename = f'claims_precip_{DATASET}.csv'
    claims.to_csv(config.output_dir / filename, index=False)


def generate_plots():
    # Load csv
    filename = f'claims_precip_{DATASET}.csv'
    claims = pd.read_csv(config.output_dir / filename)

    # Copy of the claims with positive precipitation only
    claims_pos = claims[claims['precip_max'] > 0].copy()

    # Plot a histogram of the max precipitation (all)
    filename = f'hist_precip_max_{DATASET}_all.png'
    claims['precip_max'].hist(bins=50)
    plt.title('Max precipitation intensity (all)')
    plt.xlabel('Precipitation intensity [mm/h]')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(config.output_dir / filename)
    plt.close()

    # Plot a histogram of the max precipitation (>0)
    filename = f'hist_precip_max_{DATASET}_pos.png'
    claims_pos['precip_max'].hist(bins=50)
    plt.title('Max precipitation intensity (>0)')
    plt.xlabel('Precipitation intensity [mm/h]')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(config.output_dir / filename)
    plt.close()

    # Plot a histogram of the max precipitation quantile (all)
    filename = f'hist_precip_max_q_{DATASET}_all.png'
    claims['precip_max_q'].hist(bins=50)
    plt.title('Quantiles of max precipitation intensity (all)')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(config.output_dir / filename)
    plt.close()

    # Plot a histogram of the max precipitation quantile (>0)
    filename = f'hist_precip_max_q_{DATASET}_pos.png'
    claims_pos['precip_max_q'].hist(bins=50)
    plt.title('Quantiles of max precipitation intensity (>0)')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(config.output_dir / filename)
    plt.close()

    # Plot a histogram of the time to max precipitation
    filename = f'hist_precip_dt_{DATASET}.png'
    nbins = int((PRECIP_DAYS_BEFORE + PRECIP_DAYS_AFTER + 1) * 24 / 2)
    claims['precip_dt'].hist(bins=nbins)
    plt.xlim(-(12 + PRECIP_DAYS_BEFORE * 24), 12 + PRECIP_DAYS_AFTER * 24)
    plt.title('Time to max intensity')
    plt.xlabel('Time to max intensity [h]')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(config.output_dir / filename)
    plt.close()

    # Plot a histogram of the max precipitation in 6h (all)
    filename = f'hist_precip_06h_max_{DATASET}_all.png'
    claims['precip_06h_max'].hist(bins=50)
    plt.title('Max 6-hrly precipitation total (all)')
    plt.xlabel('Precipitation total over 6h [mm]')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(config.output_dir / filename)
    plt.close()

    # Plot a histogram of the max precipitation in 6h (>0)
    filename = f'hist_precip_06h_max_{DATASET}_pos.png'
    claims_pos['precip_06h_max'].hist(bins=50)
    plt.title('Max 6-hrly precipitation total (>0)')
    plt.xlabel('Precipitation total over 6h [mm]')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(config.output_dir / filename)
    plt.close()

    # Plot a histogram of the max precipitation in 12h (all)
    filename = f'hist_precip_12h_max_{DATASET}_all.png'
    claims['precip_12h_max'].hist(bins=50)
    plt.title('Max 12-hrly precipitation total (all)')
    plt.xlabel('Precipitation total over 12h [mm]')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(config.output_dir / filename)
    plt.close()

    # Plot a histogram of the max precipitation in 12h (>0)
    filename = f'hist_precip_12h_max_{DATASET}_pos.png'
    claims_pos['precip_12h_max'].hist(bins=50)
    plt.title('Max 12-hrly precipitation total (>0)')
    plt.xlabel('Precipitation total over 12h [mm]')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(config.output_dir / filename)
    plt.close()

    # Plot a histogram of the max precipitation in 24h (all)
    filename = f'hist_precip_24h_max_{DATASET}_all.png'
    claims['precip_24h_max'].hist(bins=50)
    plt.title('Max 24-hrly precipitation total (all)')
    plt.xlabel('Precipitation total over 24h [mm]')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(config.output_dir / filename)
    plt.close()

    # Plot a histogram of the max precipitation in 24h (>0)
    filename = f'hist_precip_24h_max_{DATASET}_pos.png'
    claims_pos['precip_24h_max'].hist(bins=50)
    plt.title('Max 24-hrly precipitation total (>0)')
    plt.xlabel('Precipitation total over 24h [mm]')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(config.output_dir / filename)
    plt.close()


if __name__ == '__main__':
    main()
