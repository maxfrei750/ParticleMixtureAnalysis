from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from study_binarization_methods import RESULTS_FILE_NAME

from paddle.visualization import save_figure


def plot_binarization_methods():
    markersize = 3

    thresholds = pd.read_csv(RESULTS_FILE_NAME, index_col=0)
    # thresholds = thresholds.drop(columns=["Minimum"])  # highest thresholds (outlier)
    # thresholds = thresholds.drop(columns=["Triangle"])  # lowest thresholds (outlier)
    ax = sns.boxplot(
        data=thresholds,
        palette="viridis",
        saturation=1,
        linewidth=0.5,
        flierprops=dict(markerfacecolor="k", markeredgecolor="k", markersize=markersize),
    )
    plt.setp(ax.artists, edgecolor="black")
    plt.setp(ax.lines, color="black")
    plt.ylim([0, 1])
    plt.ylabel("Binarization threshold")
    plt.xlabel("Binarization method")

    outlier_names = ["Minimum", "Triangle"]

    for i, xticklabel in enumerate(ax.get_xticklabels()):
        if xticklabel.get_text() in outlier_names:
            xticklabel.set_color("gray")
            ax.artists[i].set_facecolor("lightgray")
            ax.artists[i].set_edgecolor("gray")

            num_lines_per_plot = 6
            start_idx = num_lines_per_plot * i
            end_idx = start_idx + num_lines_per_plot
            for line in ax.lines[start_idx:end_idx]:
                line.set_color("gray")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="d",
            color="k",
            label="Outliers",
            markerfacecolor="k",
            lw=0,
            markersize=markersize + 1,
        )
    ]
    ax.legend(handles=legend_elements)

    ax_divider = make_axes_locatable(ax)
    color_ax = ax_divider.append_axes("right", size=0.1, pad="0%")
    colorbar = plt.colorbar(cm.ScalarMappable(cmap="gray"), cax=color_ax)
    colorbar.set_ticks([])

    save_figure(".", Path(RESULTS_FILE_NAME).stem)
    plt.show()


if __name__ == "__main__":
    plot_binarization_methods()
