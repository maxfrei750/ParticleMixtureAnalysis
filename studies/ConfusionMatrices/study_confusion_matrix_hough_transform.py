from pathlib import Path

from matplotlib import pyplot as plt

from paddle.visualization import save_figure
from utilities.confusion_matrix import get_confusion_matrix_for_data_sets


def study_confusion_matrix_hough_transform():
    config_name_target = "Manual"
    config_name_prediction = "HoughTransform"

    data_root = Path("output")

    subset_target = "validation_real_merged_by_vote"
    subset_prediction = "validation_real_max"

    output_root = "studies/ConfusionMatrices/results"

    output_root = Path(output_root)
    output_root.mkdir(exist_ok=True, parents=True)

    data_root_target = data_root / config_name_target
    data_root_prediction = data_root / config_name_prediction

    confusion_matrix, figure_handle = get_confusion_matrix_for_data_sets(
        data_root_prediction, data_root_target, subset_prediction, subset_target
    )

    file_name_base = f"confusion_matrix_{config_name_prediction}"
    confusion_matrix.to_csv(output_root / f"{file_name_base}.csv")
    save_figure(output_root, file_name_base)
    plt.close()


if __name__ == "__main__":
    study_confusion_matrix_hough_transform()
