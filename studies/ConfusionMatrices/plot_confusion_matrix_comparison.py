from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from paddle.visualization import plot_confusion_matrix, save_figure


def plot_confusion_matrix_comparison():
    data_root = "results"
    upper_colorbar_limit = 1200
    input_file_names = [
        "confusion_matrix_HoughTransform.csv",
        "confusion_matrix_MultiClassSynthetic400SyntheticValidation400Detections.csv",
    ]

    data_root = Path(data_root)

    for input_file_name in input_file_names:
        file_name_base = Path(input_file_name).stem

        confusion_matrix = pd.read_csv(data_root / input_file_name, index_col=0).astype(int)
        confusion_matrix = confusion_matrix.rename(columns={"background": "vacancy"})
        confusion_matrix = confusion_matrix.rename(index={"background": "vacancy"})

        class_names = list(confusion_matrix.index)

        plot_confusion_matrix(confusion_matrix, class_names=class_names)

        plt.gca().images[0].set_clim(0, upper_colorbar_limit)

        plt.xlabel("Predicted class")
        plt.ylabel("True class")

        save_figure(data_root, file_name_base)
        plt.close()


if __name__ == "__main__":
    plot_confusion_matrix_comparison()
