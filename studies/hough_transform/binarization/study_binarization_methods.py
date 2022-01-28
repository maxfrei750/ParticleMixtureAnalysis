from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import (threshold_isodata, threshold_li, threshold_mean, threshold_minimum,
                             threshold_otsu, threshold_triangle)
from tqdm import tqdm

from paddle.data import MaskRCNNDataset
from utilities.image_processing import rgb2gray

SUBSET = "validation_real_max"
DATA_ROOT = "../../../data/complete/"

RESULT_FOLDER_NAME = "binarizations"
RESULTS_FILE_NAME = "binarization_comparison.csv"

APERTURE_MASK = np.asarray(Image.open("../aperture_mask.png").convert("1"))


def study_binarization_methods():

    data_set = MaskRCNNDataset(
        DATA_ROOT,
        subset=SUBSET,
    )

    result_folder_name = Path(RESULT_FOLDER_NAME)
    result_folder_name.mkdir(exist_ok=True, parents=True)

    thresholds = []

    for image_tensor, target in tqdm(data_set):
        image_name = target["image_name"]

        image_raw = image_tensor.numpy().transpose(1, 2, 0)
        image_grayscale = rgb2gray(image_raw)

        image_thresholds = get_thresholds(image_grayscale[APERTURE_MASK])
        image_thresholds.name = image_name

        save_image_in_results_folder(image_grayscale, f"image_{image_name}.png")

        for method_name, threshold in image_thresholds.items():
            image_binary = image_grayscale > threshold
            save_image_in_results_folder(
                image_binary, f"binarization_{image_name}_{method_name}.png"
            )

        thresholds.append(image_thresholds)

    thresholds = pd.DataFrame(thresholds)

    thresholds.to_csv(RESULTS_FILE_NAME)


def save_image_in_results_folder(image: np.ndarray, file_name: str):
    file_path = Path(RESULT_FOLDER_NAME) / file_name
    Image.fromarray(image.astype(float) * 256).convert("L").save(file_path)


def get_thresholds(image: np.ndarray) -> pd.Series:

    # Global algorithms.
    methods = {
        "ISODATA": threshold_isodata,
        "Li": threshold_li,
        "Mean": threshold_mean,
        "Minimum": threshold_minimum,
        "Otsu": threshold_otsu,
        "Triangle": threshold_triangle,
    }

    thresholds = pd.Series(index=methods.keys(), dtype=float)

    for name, thresholding_function in methods.items():
        thresholds[name] = thresholding_function(image)

    return thresholds


if __name__ == "__main__":
    study_binarization_methods()
