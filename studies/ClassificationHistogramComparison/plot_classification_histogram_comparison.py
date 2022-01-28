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


def plot_classification_histogram_comparison(
    result_file_path: AnyPath,
    colors_light: Dict[str, Tuple[float, float, float]],
    plot_file_name: str,
    darkening_factor: float = 0.75,
):
    with open(result_file_path, "rb") as file:
        results = pickle.load(file)

    colors_dark = {
        method_name: tuple(c * darkening_factor for c in color)
        for method_name, color in colors_light.items()
    }

    for method_name in tqdm(results.keys()):
        for class_name, values in results[method_name].items():
            scores = values["scores"]
            intensity_values = values["pixel_intensities"]

            if class_name == "dark":
                color = colors_dark[method_name]
            elif class_name == "light":
                color = colors_light[method_name]
            else:
                raise ValueError("Unknown class: {class_name}")

            if "manual reference" in method_name.lower() or "real" in method_name.lower():
                linestyle = "--"
            else:
                linestyle = "-"

            sns.kdeplot(
                x=intensity_values,
                weights=scores,
                markersize=0,
                color=color,
                linestyle=linestyle,
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
    PLOT_FILE_NAME = "classification_histogram_comparison"

    viridis_colors_base = get_viridis_colors(2)
    COLORS_LIGHT = {
        "Manual reference": (0.5, 0.5, 0.5),
        "Proposed method": viridis_colors_base[0],
        "Hough transform": viridis_colors_base[1],
    }

    plot_classification_histogram_comparison(
        RESULT_FILE_NAME, colors_light=COLORS_LIGHT, plot_file_name=PLOT_FILE_NAME
    )
