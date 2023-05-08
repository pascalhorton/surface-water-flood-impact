import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
from pathlib import Path


def plot_heatmap_differences(data, total, labels, title, dir_output=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data)

    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    threshold = im.norm(data.max()) / 2.
    kw = dict(ha='center', va='center')
    text_colors = ('white', 'black')

    for i in range(len(labels)):
        for j in range(len(labels)):
            kw.update(color=text_colors[int(im.norm(data[i, j]) > threshold)])
            if i > j:
                pc = 100 * data[i, j] / total
                ax.text(j, i, f'{pc:.2f}%', **kw)
            elif j > i:
                ax.text(j, i, f'{int(data[i, j])}', **kw)

    ax.set_title(title)
    fig.tight_layout()

    _save_or_show(dir_output, title)


def plot_histogram_time_difference(claims, field_name, title, dir_output=None):
    hist_bins = claims[field_name].max() - claims[field_name].min() + 1
    hist_range = (claims[field_name].min() - 0.5, claims[field_name].max() + 0.5)
    counts, bins = np.histogram(claims[field_name], bins=hist_bins, range=hist_range)
    plt.stairs(counts, bins)
    plt.title(title)
    plt.xlabel("Difference in days")
    plt.ylabel("Count")
    plt.tight_layout()

    _save_or_show(dir_output, title)


def plot_claim_events_timeseries(window_days, precip, claim_1, label_1, claim_2=None,
                                 label_2=None, title=None, dir_output=None):
    cid = claim_1.cid
    claim_date = claim_1.date_claim

    # Get first event data
    e_1_dur = (claim_1.e_end - claim_1.e_start).total_seconds() / 3600
    e_1_dates = [claim_1.e_start + timedelta(hours=x) for x in range(int(e_1_dur) + 1)]
    precip_1 = precip.get_time_series(cid, claim_1.e_start, claim_1.e_end)

    # Get second event data
    e_2_dates = None
    precip_2 = None
    if claim_2 is not None:
        e_2_dur = (claim_2.e_end - claim_2.e_start).total_seconds() / 3600
        e_2_dates = [claim_2.e_start + timedelta(hours=x) for x in range(int(e_2_dur) + 1)]
        precip_2 = precip.get_time_series(cid, claim_2.e_start, claim_2.e_end)

    # Get full window data
    delta_days_max = (max(window_days) - 1) / 2
    timedelta_max = timedelta(days=delta_days_max)
    win_start = datetime.combine(claim_date - timedelta_max, datetime.min.time())
    win_end = datetime.combine(claim_date + timedelta_max, datetime.max.time())
    precip_win = precip.get_time_series(cid, win_start, win_end, size=3)
    precip_win_orig = precip.get_time_series(cid, win_start, win_end, size=1)
    dates_win = [win_start + timedelta(hours=x) for x in range(len(precip_win))]

    # Plot precipitation
    fig, axs = plt.subplots(figsize=(12, 4))
    plt.plot(dates_win, precip_win_orig, label='not smoothed', linewidth=1,
             color='0.3', linestyle='dotted')
    plt.plot(dates_win, precip_win, linewidth=1, color='0.1')
    plt.plot(e_1_dates, precip_1, label=label_1, linewidth=2)
    if claim_2 is not None:
        plt.plot(e_2_dates, precip_2, label=label_2, linewidth=2)

    # Add vertical span corresponding to the time windows
    for i_win, window in enumerate(window_days):
        delta_days = (window - 1) / 2
        win_start = datetime.combine(
            claim_date - timedelta(days=delta_days), datetime.min.time())
        win_end = datetime.combine(
            claim_date + timedelta(days=delta_days), datetime.max.time())
        alpha = 0.1 + 0.4 * i_win / len(window_days)
        axs.axvspan(win_start, win_end, facecolor='gray', alpha=alpha)

    plt.legend(loc='upper right')
    plt.ylabel("Precipitation [mm/h]")
    if title is not None:
        plt.title(title)
    plt.tight_layout()

    filename = f"{claim_date} {cid}"

    _save_or_show(dir_output, filename)


def _save_or_show(dir_output, title):
    if dir_output is not None:
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)
        filename = re.sub(r'\W+', '', title.replace(' ', '_'))
        plt.savefig(dir_output / (filename + '.png'), dpi=600)
        plt.savefig(dir_output / (filename + '.pdf'), dpi=600)
        plt.close()
    else:
        plt.show()
