from pathlib import Path

import numpy as np
import pandas as pd


def plot_accuracy_comparison():
    data_root = "results"
    input_file_names = [
        "confusion_matrix_HoughTransform.csv",
        "confusion_matrix_MultiClassSynthetic400SyntheticValidation400Detections.csv",
    ]

    method_labels = ["Hough transform", "Proposed method"]

    data_root = Path(data_root)

    for input_file_name, method_label in zip(input_file_names, method_labels):

        confusion_matrix = pd.read_csv(data_root / input_file_name, index_col=0).astype(int)

        accuracy = calculate_accuracy(confusion_matrix)

        confusion_matrix_background_object = merge_classes(
            confusion_matrix, class_name_a="dark", class_name_b="light", class_name_new="object"
        )
        accuracy_background_object = calculate_accuracy(confusion_matrix_background_object)

        confusion_matrix_dark_light = drop_class(confusion_matrix, class_name="background")

        accuracy_dark_light = calculate_accuracy(confusion_matrix_dark_light)

        print("\033[4m" + method_label + "\033[0m")
        print(f"{accuracy=:.1%}")
        print(f"{accuracy_background_object=:.1%}")
        print(f"{accuracy_dark_light=:.1%}")
        print("")

        # save_figure(data_root, file_name_base)
        # plt.close()


def drop_class(confusion_matrix, class_name):
    confusion_matrix = confusion_matrix.copy()
    confusion_matrix = confusion_matrix.drop(class_name, axis=1)
    confusion_matrix = confusion_matrix.drop(class_name, axis=0)
    return confusion_matrix


def merge_classes(confusion_matrix, class_name_a, class_name_b, class_name_new):
    confusion_matrix = confusion_matrix.copy()
    confusion_matrix[class_name_new] = (
        confusion_matrix[class_name_a] + confusion_matrix[class_name_b]
    )
    del confusion_matrix[class_name_a]
    del confusion_matrix[class_name_b]
    confusion_matrix = (
        confusion_matrix.reset_index()
        .replace({"index": {class_name_a: class_name_b}})
        .groupby("index", sort=False)
        .sum()
        .rename(index={class_name_b: class_name_new})
    )
    return confusion_matrix


def calculate_accuracy(confusion_matrix):
    num_predictions = confusion_matrix.sum().sum()
    num_true = np.trace(confusion_matrix)
    accuracy = num_true / num_predictions
    return accuracy


if __name__ == "__main__":
    plot_accuracy_comparison()
