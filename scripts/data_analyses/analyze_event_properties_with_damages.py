"""
This script analyzes the properties of the precipitation events with and without claims.
"""

from swafi.config import Config
from swafi.events import load_events_from_pickle
import matplotlib.pyplot as plt
import numpy as np

CONFIG = Config()

DATASET = 'mobiliar'  # 'mobiliar' or 'gvz'
LABEL_EVENT_FILE = 'default_occurrence'
DO_PRINT = True
PLOT_SYNTHESIS = True
PLOT_INDIVIDUAL = False

config = Config(output_dir='analysis_event_properties_with_damages')
output_dir = config.output_dir


def main():
    # Load events
    events_filename = f'events_{DATASET}_with_target_{LABEL_EVENT_FILE}.pickle'
    events = load_events_from_pickle(filename=events_filename)
    events_df = events.events

    # Split the events with damages vs those without damage
    with_claims = events_df[events_df['target'] > 0]
    without_claims = events_df[events_df['target'] == 0]

    if PLOT_SYNTHESIS:
        cmap = plt.get_cmap('viridis')
        fig, ax = plt.subplots(4, 5, figsize=(16, 16))

        plot_histo(with_claims, without_claims, 'i_max', max_val=60,
                   ax_1=ax[0, 0], ax_2=ax[1, 0], title='Maximum intensity [mm/h]',
                   color_1=cmap(0.3), color_2=cmap(0.7), show_ylabel=True)
        plot_histo(with_claims, without_claims, 'i_mean', max_val=25,
                   ax_1=ax[0, 1], ax_2=ax[1, 1], title='Mean intensity [mm/h]',
                   color_1=cmap(0.3), color_2=cmap(0.7), show_ylabel=False)
        plot_histo(with_claims, without_claims, 'p_sum', max_val=160,
                   ax_1=ax[0, 2], ax_2=ax[1, 2], title='Precipitation sum [mm]',
                   color_1=cmap(0.3), color_2=cmap(0.7), show_ylabel=False)
        plot_histo(with_claims, without_claims, 'api', max_val=100,
                   ax_1=ax[0, 3], ax_2=ax[1, 3], title='API [mm]',
                   color_1=cmap(0.3), color_2=cmap(0.7), show_ylabel=False)
        plot_histo(with_claims, without_claims, 'duration', max_val=120,
                   n_bins=60, ax_1=ax[0, 4], ax_2=ax[1, 4], title='Duration [h]',
                   color_1=cmap(0.3), color_2=cmap(0.7), show_ylabel=False)

        plot_histo(with_claims, without_claims, 'i_max_q', max_val=1,
                   ax_1=ax[2, 0], ax_2=ax[3, 0], title='Quantiles max intensity',
                   color_1=cmap(0.3), color_2=cmap(0.7), show_ylabel=True)
        plot_histo(with_claims, without_claims, 'i_mean_q', max_val=1,
                   ax_1=ax[2, 1], ax_2=ax[3, 1], title='Quantiles mean intensity',
                   color_1=cmap(0.3), color_2=cmap(0.7), show_ylabel=False)
        plot_histo(with_claims, without_claims, 'p_sum_q', max_val=1,
                   ax_1=ax[2, 2], ax_2=ax[3, 2], title='Quantiles precip. sum',
                   color_1=cmap(0.3), color_2=cmap(0.7), show_ylabel=False)
        plot_histo(with_claims, without_claims, 'api_q', max_val=1,
                   ax_1=ax[2, 3], ax_2=ax[3, 3], title='Quantiles API',
                   color_1=cmap(0.3), color_2=cmap(0.7), show_ylabel=False)
        plot_histo(with_claims, without_claims, 'duration_q', max_val=1,
                   ax_1=ax[2, 4], ax_2=ax[3, 4], title='Quantiles duration',
                   color_1=cmap(0.3), color_2=cmap(0.7), show_ylabel=False)

        plt.tight_layout()
        plt.savefig(output_dir / 'event_damage_properties.png')
        plt.savefig(output_dir / 'event_damage_properties.pdf')

        plt.close()

    # Plot the distributions
    if PLOT_INDIVIDUAL:
        plot_histo(with_claims, without_claims, 'i_max', max_val=60)
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
        plot_histo(with_claims, without_claims, 'duration', max_val=150)
        plot_histo(with_claims, without_claims, 'duration', log_scale=True)
        plot_histo(with_claims, without_claims, 'api', max_val=100)
        plot_histo(with_claims, without_claims, 'api', log_scale=True)
        plot_histo(with_claims, without_claims, 'api_q')
        plot_histo(with_claims, without_claims, 'api_q', log_scale=True)
        plot_histo(with_claims, without_claims, 'nb_contracts')
        plot_histo(with_claims, without_claims, 'nb_contracts', log_scale=True)

    print("Done.")


def get_common_bins(df1, df2, n_bins=50, max_val=None):
    if max_val is None:
        max_val = max(df1.max(), df2.max())

    # Ensure that both bin widths are the same
    return np.linspace(min(df1.min(), df2.min()), max_val, n_bins)


def plot_histo(df_claims, df_no_claims, param, log_scale=False, max_val=None,
               n_bins=50, ax_1=None, ax_2=None, title=None, color_1=None, color_2=None,
               show_ylabel=True):

    common_bins = get_common_bins(df_claims[param], df_no_claims[param], n_bins, max_val)

    if not ax_1:
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        ax_1 = axs[0]
        ax_2 = axs[1]

    ax_1.hist(df_claims[param], bins=common_bins, color=color_1)
    ax_2.hist(df_no_claims[param], bins=common_bins, color=color_2)

    if show_ylabel:
        ax_1.set_ylabel('Number of events')
        ax_2.set_ylabel('Number of events')

    if not ax_1:
        plt.xlabel(param)

    if max_val is not None:
        ax_1.set_xlim(0, max_val)
        ax_2.set_xlim(0, max_val)
    else:
        ax_1.set_xlim(0, None)
        ax_2.set_xlim(0, None)

    if title is not None:
        ax_1.set_title(title)
        ax_2.set_title(title)
    else:
        if log_scale:
            ax_1.set_title(f'Distribution ({param}) for events with claims (log scale)')
            ax_2.set_title(f'Distribution ({param}) for events without claims (log scale)')
        else:
            ax_1.set_title(f'Distribution ({param}) for events with claims')
            ax_2.set_title(f'Distribution ({param}) for events without claims')

    if log_scale:
        ax_1.set_yscale('log')
        ax_2.set_yscale('log')

    if ax_1 is not None:
        return

    plt.tight_layout()
    if DO_PRINT:
        plt.savefig(f'{output_dir}/histo_{param}_{"log" if log_scale else ""}.png')
        plt.savefig(f'{output_dir}/histo_{param}_{"log" if log_scale else ""}.pdf')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    main()
