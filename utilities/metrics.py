import torch
from tqdm import tqdm


def apply_metric_to_data_sets(data_set_prediction, data_set_target, metric):
    assert (
        data_set_prediction.class_names == data_set_target.class_names
    ), "Target and prediction data sets must use the same classes."
    for (_, target), (_, prediction) in tqdm(zip(data_set_target, data_set_prediction)):
        assert (
            target["image_name"] == prediction["image_name"]
        ), "Prediction and target must belong to the same input image."

        # Match the output format of the Mask R-CNN model.
        prediction["masks"] = prediction["masks"].float()
        prediction["masks"] = torch.unsqueeze(prediction["masks"], dim=1)

        metric.update([prediction], (target,))
