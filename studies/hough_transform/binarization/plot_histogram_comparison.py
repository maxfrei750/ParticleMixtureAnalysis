import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from tqdm import tqdm

from paddle.data import MaskRCNNDataset
from paddle.visualization import get_viridis_colors, save_figure
from utilities.image_processing import rgb2gray

SUBSET = "validation_real_max"
DATA_ROOT = "../../../data/complete/"

RESULTS_FILE_NAME_BASE = "histogram_comparison"

APERTURE_MASK = np.asarray(Image.open("../aperture_mask.png").convert("1"))


def plot_histogram_comparison():
    sns.color_palette("viridis", as_cmap=True)

    data_set = MaskRCNNDataset(
        DATA_ROOT,
        subset=SUBSET,
    )

    num_images = len(data_set)
    colors = get_viridis_colors(num_images)

    for color, (image_tensor, target) in tqdm(zip(colors, data_set)):
        image_raw = image_tensor.numpy().transpose(1, 2, 0)
        image_grayscale = rgb2gray(image_raw)

        sns.kdeplot(image_grayscale[APERTURE_MASK], markersize=0, linewidth=0.5, color=color)

    plt.xlim([0, 1])
    plt.ylabel("Probability density")
    plt.xlabel("Pixel intensity")

    ax = plt.gca()
    ax_divider = make_axes_locatable(ax)
    color_ax = ax_divider.append_axes("top", size=0.1, pad="0%")
    colorbar = plt.colorbar(cm.ScalarMappable(cmap="gray"), cax=color_ax, orientation="horizontal")
    colorbar.set_ticks([])

    save_figure(".", RESULTS_FILE_NAME_BASE)
    plt.show()


if __name__ == "__main__":
    plot_histogram_comparison()
