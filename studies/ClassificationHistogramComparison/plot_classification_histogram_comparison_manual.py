import pickle
from typing import Dict, Tuple

import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from study_classification_histogram_comparison import RESULT_FILE_NAME
from tqdm import tqdm

from paddle.custom_types import AnyPath
from paddle.visualization import get_viridis_colors, save_figure


def plot_classification_histogram_comparison_manual(
    result_file_path: AnyPath,
    plot_file_name: str,
):
    with open(result_file_path, "rb") as file:
        results = pickle.load(file)

    viridis_colors_base = get_viridis_colors(2)

    for method_name in tqdm(results.keys()):
        if "manual reference" not in method_name.lower():
            continue

        for class_name, values in results[method_name].items():
            scores = values["scores"]
            intensity_values = values["pixel_intensities"]

            if class_name == "dark":
                color = viridis_colors_base[0]
            elif class_name == "light":
                color = viridis_colors_base[1]
            else:
                raise ValueError("Unknown class: {class_name}")

            sns.kdeplot(
                x=intensity_values,
                weights=scores,
                markersize=0,
                color=color,
                linestyle="-",
                label=f"{method_name} ({class_name})",
            )

    plt.legend()

    plt.xlim([0, 1])
    plt.ylabel("Probability density")
    plt.xlabel("Pixel intensity")

    ax = plt.gca()
    ax_divider = make_axes_locatable(ax)
    color_ax = ax_divider.append_axes("top", size=0.1, pad="0%")
    colorbar = plt.colorbar(cm.ScalarMappable(cmap="gray"), cax=color_ax, orientation="horizontal")
    colorbar.set_ticks([])

    save_figure(".", plot_file_name)
    plt.show()


if __name__ == "__main__":
    PLOT_FILE_NAME = "classification_histogram_comparison_manual"

    plot_classification_histogram_comparison_manual(RESULT_FILE_NAME, plot_file_name=PLOT_FILE_NAME)
