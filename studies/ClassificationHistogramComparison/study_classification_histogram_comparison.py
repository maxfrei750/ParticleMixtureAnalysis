import pickle
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from paddle.custom_types import AnyPath
from paddle.data import MaskRCNNDataset

DATA_ROOT = Path("../../output/")
RESULT_FILE_NAME = "results.pickle"

DATA_SETS = {
    "Manual reference": MaskRCNNDataset(
        DATA_ROOT / "Manual",
        subset="validation_real_merged_by_vote",
    ),
    "Proposed method": MaskRCNNDataset(
        DATA_ROOT / "MultiClassSynthetic400SyntheticValidation400Detections",
        subset="validation_real_max",
    ),
    "Hough transform": MaskRCNNDataset(
        DATA_ROOT / "HoughTransform",
        subset="validation_real_max",
    ),
}


def study_classification_histogram_comparison(
    data_sets: Dict[str, MaskRCNNDataset], result_file_path: AnyPath
):
    """Study how the histograms of different classes compare for different methods.

    :param data_sets: dictionary of datasets with method names as keys
    :param result_file_path: path of an output pickle file
    """
    class_names = ["dark", "light"]

    results = {
        method_name: {
            class_name: {"pixel_intensities": [], "scores": []} for class_name in class_names
        }
        for method_name in data_sets.keys()
    }

    for method_name, data_set in data_sets.items():
        for image, target in tqdm(data_set, desc=f"Analyzing {method_name}"):
            image = image.mean(axis=0)

            for mask, label, score in zip(target["masks"], target["labels"], target["scores"]):
                class_name = data_set.map_label_to_class_name[int(label)]
                pixel_intensities = image[mask.bool()]

                scores = torch.ones_like(pixel_intensities) * score
                results[method_name][class_name]["pixel_intensities"].append(pixel_intensities)
                results[method_name][class_name]["scores"].append(scores)

    # Flatten result dict
    for method_name in results.keys():
        for class_name in results[method_name].keys():
            for key, value in results[method_name][class_name].items():
                results[method_name][class_name][key] = list(torch.cat(value).numpy())

    with open(result_file_path, "wb") as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    study_classification_histogram_comparison(DATA_SETS, RESULT_FILE_NAME)
