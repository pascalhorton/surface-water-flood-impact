"""
This script analyzes the distribution of the number of contracts and claims per cell.
"""

from swafi.config import Config
from swafi.damages_mobiliar import DamagesMobiliar
from swafi.damages_gvz import DamagesGvz
import pandas as pd
import matplotlib.pyplot as plt

config = Config(output_dir='analysis_damage_distribution')
output_dir = config.output_dir

PICKLES_DIR = config.get('PICKLES_DIR')
DATASET = 'mobiliar'  # 'mobiliar' or 'gvz'

if DATASET == 'mobiliar':
    EXPOSURE_CATEGORIES = ['external']
    CLAIM_CATEGORIES = ['external', 'pluvial']
elif DATASET == 'gvz':
    EXPOSURE_CATEGORIES = ['all_buildings']
    CLAIM_CATEGORIES = ['A', 'B']



def main():
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

    # Plot the monthly distribution of the number of claims for the different categories
    df_claims_month = damages.claims
    df_claims_month['date_claim'] = pd.to_datetime(
        df_claims_month['date_claim'], errors='coerce')
    df_claims_month['month'] = df_claims_month['date_claim'].dt.month
    df_claims_month = df_claims_month.drop(
        columns=['date_claim', 'mask_index', 'cid', 'x', 'y'])
    df_claims_month_sum = df_claims_month.groupby('month').sum()

    # Plot the monthly distribution of the total # of claims for different categories
    for category in damages.claim_categories:
        sum_claims = df_claims_month_sum[category].sum()
        plt.figure(figsize=(8, 4))
        plt.title(f'Monthly distribution of total claims for category {category} '
                  f'(total: {sum_claims})')
        plt.xlabel('Month')
        plt.ylabel('Percentage of claims [%]')
        nb_annual_claims = df_claims_month_sum[category] / sum_claims
        plt.bar(df_claims_month_sum.index, 100 * nb_annual_claims)
        plt.xticks(range(1, 13))
        plt.tight_layout()
        plt.savefig(output_dir / f'monthly_distribution_tot_claims_{category}.png')
        plt.savefig(output_dir / f'monthly_distribution_tot_claims_{category}.pdf')

    # Plot the monthly distribution of the mean # of claims for different categories
    for category in damages.claim_categories:
        df_claims_month_mean = df_claims_month[df_claims_month[category] > 0]
        df_claims_month_mean = df_claims_month_mean.groupby('month').mean()
        plt.figure(figsize=(8, 4))
        plt.title(f'Monthly distribution of # of claims for category {category}')
        plt.xlabel('Month')
        plt.ylabel('Mean number of claims')
        plt.bar(df_claims_month_mean.index, df_claims_month_mean[category])
        plt.xticks(range(1, 13))
        plt.tight_layout()
        plt.savefig(output_dir / f'monthly_distribution_nb_claims_{category}.png')
        plt.savefig(output_dir / f'monthly_distribution_nb_claims_{category}.pdf')



    damages.select_categories_type(EXPOSURE_CATEGORIES, CLAIM_CATEGORIES)

    df_contracts = damages.exposure
    df_contracts = df_contracts[['mask_index', 'selection', 'cid']]

    # Average the number of annual contracts per location
    df_contracts = df_contracts.groupby('mask_index').mean()

    # Plot the histogram of the number of contracts per cell
    plt.figure()
    plt.title('Histogram of the number of contracts per cell')
    plt.xlabel('Number of contracts')
    plt.ylabel('Number of cells')
    plt.hist(df_contracts['selection'], bins=100)
    plt.yscale('log')
    plt.xlim(0, None)
    plt.tight_layout()
    plt.savefig(output_dir / 'histogram_contracts.png')
    plt.savefig(output_dir / 'histogram_contracts.pdf')

    df_claims = damages.claims
    df_claims = df_claims[['mask_index', 'selection']]

    # Sum the number of claims per location and divide by the number of years
    df_claims = df_claims.groupby('mask_index').sum()
    n_years = damages.year_end - damages.year_start + 1
    df_claims['selection'] = df_claims['selection'] / n_years

    # Plot the histogram of the number of claims per cell
    plt.figure()
    plt.title('Histogram of the number of annual claims per cell')
    plt.xlabel('Number of claims')
    plt.ylabel('Number of cells')
    plt.hist(df_claims['selection'], bins=50)
    plt.yscale('log')
    plt.xlim(0, None)
    plt.tight_layout()
    plt.savefig(output_dir / 'histogram_claims.png')
    plt.savefig(output_dir / 'histogram_claims.pdf')

    # Merge the contracts and claims dataframes on the index
    df_merged = df_contracts.merge(df_claims, left_index=True,
                                   right_index=True, how='left')

    # Rename the columns
    df_merged.columns = ['contracts', 'cid', 'claims']

    # Replace nan values with 0
    df_merged.fillna(0, inplace=True)

    # Plot the relationship between the number of contracts and the number of claims
    plt.figure()
    plt.title('Relationship between the number of contracts and the claims')
    plt.xlabel('Number of contracts (mean per cell)')
    plt.ylabel('Mean number of annual claims (sum per cell)')
    plt.scatter(df_merged['contracts'], df_merged['claims'],
                facecolors='none', edgecolors='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_contracts_claims.png')
    plt.savefig(output_dir / 'scatter_contracts_claims.pdf')


if __name__ == '__main__':
    main()
