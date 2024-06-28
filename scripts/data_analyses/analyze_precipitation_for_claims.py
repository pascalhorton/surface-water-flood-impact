"""
This script analyzes the precipitation data characteristics for each claim.
"""
import pandas as pd

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
    # Load CombiPrecip files
    precip = CombiPrecip()
    precip.load_data(config.get('DIR_PRECIP'))
    precip.compute_quantiles()

    # Load the damage data
    if DATASET == 'mobiliar':
        damages = DamagesMobiliar(dir_exposure=config.get('DIR_EXPOSURE_MOBILIAR'),
                                  dir_claims=config.get('DIR_CLAIMS_MOBILIAR'),
                                  year_start=config.get('YEAR_START_MOBILIAR'),
                                  year_end=config.get('YEAR_END_MOBILIAR'))
    elif DATASET == 'gvz':
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

    # Add columns to the claims dataframe
    claims['precip_max'] = None
    claims['precip_dt'] = None

    for claim in claims:
        cid = claim['cid']
        date_claim = claim['date_claim']
        start = date_claim - pd.Timedelta(days=PRECIP_DAYS_BEFORE)
        end = date_claim + pd.Timedelta(days=PRECIP_DAYS_AFTER)
        precip_ts = precip.get_time_series(cid, start, end)

        # Compute the max precipitation
        claims['precip_max'] = precip_ts.max()

        # Compute centrality of the max precipitation
        idx_max = precip_ts.argmax()
        claims['precip_dt'] = idx_max - len(precip_ts) // 2





if __name__ == '__main__':
    main()
