"""
This script analyzes the distribution of the number of contracts and claims per cell.
"""

from swafi.config import Config
from swafi.events import load_events_from_pickle
import matplotlib.pyplot as plt
import numpy as np

CONFIG = Config()

LABEL_EVENT_FILE = 'original_w_prior_pluvial_occurrence'
DO_PRINT = True

config = Config(output_dir='analysis_event_properties_with_damages')
output_dir = config.output_dir


def main():
    # Load events
    events_filename = f'events_with_target_values_{LABEL_EVENT_FILE}.pickle'
    events = load_events_from_pickle(filename=events_filename)
    events_df = events.events

    # Split the events with damages vs those without damage
    with_claims = events_df[events_df['target'] == 1]
    without_claims = events_df[events_df['target'] == 0]

    # Plot the distributions
    plot_histo(with_claims, without_claims, 'i_max')
    plot_histo(with_claims, without_claims, 'i_max', log_scale=True)
    plot_histo(with_claims, without_claims, 'i_max_q')
    plot_histo(with_claims, without_claims, 'i_max_q', log_scale=True)
    plot_histo(with_claims, without_claims, 'i_mean')
    plot_histo(with_claims, without_claims, 'i_mean', log_scale=True)
    plot_histo(with_claims, without_claims, 'i_mean_q')
    plot_histo(with_claims, without_claims, 'i_mean_q', log_scale=True)
    plot_histo(with_claims, without_claims, 'i_sd')
    plot_histo(with_claims, without_claims, 'i_sd', log_scale=True)
    plot_histo(with_claims, without_claims, 'i_sd_q')
    plot_histo(with_claims, without_claims, 'i_sd_q', log_scale=True)
    plot_histo(with_claims, without_claims, 'p_sum')
    plot_histo(with_claims, without_claims, 'p_sum', log_scale=True)
    plot_histo(with_claims, without_claims, 'p_sum_q')
    plot_histo(with_claims, without_claims, 'p_sum_q', log_scale=True)
    plot_histo(with_claims, without_claims, 'e_tot')
    plot_histo(with_claims, without_claims, 'e_tot', log_scale=True)
    plot_histo(with_claims, without_claims, 'apireg')
    plot_histo(with_claims, without_claims, 'apireg', log_scale=True)
    plot_histo(with_claims, without_claims, 'apireg_q')
    plot_histo(with_claims, without_claims, 'apireg_q', log_scale=True)
    plot_histo(with_claims, without_claims, 'nb_contracts')
    plot_histo(with_claims, without_claims, 'nb_contracts', log_scale=True)

    print("Done.")


def get_common_bins(df1, df2, n_bins=100):
    # Ensure that both bin widths are the same
    common_bins = np.linspace(
        min(df1.min(), df2.min()),
        max(df1.max(), df2.max()), 100)

    return common_bins


def plot_histo(df_claims, df_no_claims, param, log_scale=False):
    common_bins = get_common_bins(df_claims[param], df_no_claims[param], 100)
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    plt.xlabel(param)
    plt.ylabel('Number of events')
    axs[0].hist(df_claims[param], bins=common_bins)
    axs[0].set_ylabel('Number of events')
    axs[1].hist(df_no_claims[param], bins=common_bins)
    axs[1].set_ylabel('Number of events')
    if log_scale:
        axs[0].set_title(f'Distribution ({param}) for events with claims (log scale)')
        axs[1].set_title(f'Distribution ({param}) for events without claims (log scale)')
    else:
        axs[0].set_title(f'Distribution ({param}) for events with claims')
        axs[1].set_title(f'Distribution ({param}) for events without claims')
    if log_scale:
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
    plt.tight_layout()
    if DO_PRINT:
        plt.savefig(f'{output_dir}/histo_{param}_{"log" if log_scale else ""}.png')
        plt.savefig(f'{output_dir}/histo_{param}_{"log" if log_scale else ""}.pdf')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    main()
