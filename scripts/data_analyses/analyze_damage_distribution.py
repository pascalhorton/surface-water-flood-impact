"""
This script analyzes the distribution of the number of contracts and claims per cell.
"""

from swafi.config import Config
from swafi.damages import Damages
import matplotlib.pyplot as plt

CONFIG = Config()

DAMAGE_CATEGORIES = ['external', 'pluvial']
PICKLES_DIR = CONFIG.get('PICKLES_DIR')

config = Config(output_dir='analysis_damage_distribution')
output_dir = config.output_dir


def main():
    damages = Damages(dir_contracts=CONFIG.get('DIR_CONTRACTS'),
                      dir_claims=CONFIG.get('DIR_CLAIMS'))
    damages.select_categories_type(DAMAGE_CATEGORIES)

    df_contracts = damages.contracts
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

    # Sum the number of annual claims per location
    df_claims = df_claims.groupby('mask_index').sum()

    # Plot the histogram of the number of claims per cell
    plt.figure()
    plt.title('Histogram of the number of claims per cell')
    plt.xlabel('Number of claims')
    plt.ylabel('Number of cells')
    plt.hist(df_claims['selection'], bins=50)
    plt.yscale('log')
    plt.xlim(0, None)
    plt.tight_layout()
    plt.savefig(output_dir / 'histogram_claims.png')
    plt.savefig(output_dir / 'histogram_claims.pdf')

    # Merge the contracts and claims dataframes on the index
    df_merged = df_contracts.merge(df_claims, left_index=True, right_index=True, how='left')

    # Rename the columns
    df_merged.columns = ['contracts', 'cid', 'claims']

    # Replace nan values with 0
    df_merged.fillna(0, inplace=True)

    # Plot the relationship between the number of contracts and the number of claims
    plt.figure()
    plt.title('Relationship between the number of contracts and the claims')
    plt.xlabel('Number of contracts (mean per cell)')
    plt.ylabel('Number of claims (sum per cell)')
    plt.scatter(df_merged['contracts'], df_merged['claims'],
                facecolors='none', edgecolors='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_contracts_claims.png')
    plt.savefig(output_dir / 'scatter_contracts_claims.pdf')

    # Plot the number of claims for different number of contracts
    contracts_ranges = [(0, 2), (0, 5), (0, 10), (10, 100), (100, 9000)]
    for cont_range in contracts_ranges:
        plt.figure()
        plt.title(f'Number of claims for {cont_range[0]} < contracts < {cont_range[1]}')
        plt.xlabel('Number of claims')
        plt.ylabel('Number of cells')
        plt.hist(df_merged[(df_merged['contracts'] > cont_range[0]) &
                           (df_merged['contracts'] < cont_range[1])]['claims'], bins=50)
        plt.yscale('log')
        plt.xlim(0, None)
        plt.tight_layout()
        plt.savefig(output_dir / f'histogram_claims_{cont_range[0]}_{cont_range[1]}.png')
        plt.savefig(output_dir / f'histogram_claims_{cont_range[0]}_{cont_range[1]}.pdf')


if __name__ == '__main__':
    main()
