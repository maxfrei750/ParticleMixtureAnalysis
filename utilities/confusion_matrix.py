from pathlib import Path

import pandas as pd

from paddle.data import MaskRCNNDataset
from paddle.metrics import ConfusionMatrix
from utilities.metrics import apply_metric_to_data_sets


def get_confusion_matrix_for_data_sets(
    data_root_prediction, data_root_target, subset_prediction, subset_target
):
    data_root_target = Path(data_root_target)
    data_root_prediction = Path(data_root_prediction)

    data_set_target = MaskRCNNDataset(
        data_root_target,
        subset=subset_target,
    )
    data_set_prediction = MaskRCNNDataset(
        data_root_prediction,
        subset=subset_prediction,
    )
    metric = ConfusionMatrix(data_set_target.num_classes, iou_type="mask", iou_threshold=0.5)
    apply_metric_to_data_sets(data_set_prediction, data_set_target, metric)
    confusion_matrix = pd.DataFrame(
        metric.compute().numpy(),
        index=data_set_target.class_names,
        columns=data_set_target.class_names,
    )
    figure_handle = metric.plot(data_set_target.class_names)
    return confusion_matrix, figure_handle
