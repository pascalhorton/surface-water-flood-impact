import re
import numpy as np
import matplotlib.pyplot as plt
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


def _save_or_show(dir_output, title):
    if dir_output is not None:
        dir_output = Path(dir_output)
        filename = re.sub(r'\W+', '', title.replace(' ', '_'))
        plt.savefig(dir_output / (filename + '.png'), dpi=600)
        plt.savefig(dir_output / (filename + '.pdf'), dpi=600)
        plt.close()
    else:
        plt.show()
