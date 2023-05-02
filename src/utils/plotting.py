import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap_differences(data, total, labels, title):
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
    plt.show()
