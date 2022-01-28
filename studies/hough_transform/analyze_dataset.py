from pathlib import Path
from typing import Optional, Tuple

import fire
import numpy as np
from numpy.typing import ArrayLike
from PIL import Image as PILImage
from skimage import exposure, img_as_ubyte
from tqdm import tqdm

from paddle.custom_types import AnyPath
from paddle.data import MaskRCNNDataset
from utilities.circle_detection_hough import circle_detection_hough, circle_detections_to_masks
from utilities.image_processing import apply_mean_threshold, rgb2gray

APERTURE_MASK_PATH = Path(__file__).parent / "aperture_mask.png"

Image = ArrayLike
Mask = ArrayLike
CircleDetectionResult = Tuple[ArrayLike, ArrayLike, ArrayLike]


def test_hough_transform_on_dataset(
    subset: str,
    data_root: Optional[AnyPath] = "data/complete",
    output_root: Optional[AnyPath] = "output",
):
    """Analyses the samples of a dataset using Hough transformation.

    :param subset: Name of the subset to use.
    :param data_root: Path of the data set folder, holding the subsets.
    :param output_root: Root directory for output files.
    """
    data_root = Path(data_root)
    output_root = Path(output_root) / "HoughTransform" / subset

    output_root.mkdir(parents=True, exist_ok=True)

    aperture_mask = PILImage.open(APERTURE_MASK_PATH).convert("1")
    aperture_mask = np.array(aperture_mask)

    data_set = MaskRCNNDataset(
        data_root,
        subset=subset,
    )

    for image_tensor, target in tqdm(data_set):

        image_name = target["image_name"]

        image_raw = image_tensor.numpy().transpose(1, 2, 0)
        image_grayscale = rgb2gray(image_raw)

        # Save image.
        output_image_path = output_root / f"image_{image_name}.png"
        PILImage.fromarray(img_as_ubyte(image_grayscale)).save(output_image_path)

        image_binary = apply_mean_threshold(image_grayscale, aperture_mask)

        # Dark particles
        class_name = "dark"
        particle_mask_dark = aperture_mask & ~image_binary

        analyze_image_hough_transform(
            image_grayscale, image_name, particle_mask_dark, class_name, output_root
        )

        # Light particles
        class_name = "light"
        particle_mask_light = aperture_mask & image_binary

        analyze_image_hough_transform(
            image_grayscale, image_name, particle_mask_light, class_name, output_root
        )


def analyze_image_hough_transform(
    image_grayscale, image_name, particle_mask, class_name, output_root
):
    class_folder_path = output_root / class_name
    class_folder_path.mkdir(exist_ok=True, parents=True)
    image_equalized = exposure.equalize_hist(image_grayscale, mask=particle_mask)
    centroids_x, centroids_y, radii = circle_detection_hough(
        image_equalized, radius_min=20, radius_max=75, mask=particle_mask
    )
    masks = circle_detections_to_masks(image_grayscale, centroids_x, centroids_y, radii)
    for mask_id, mask in enumerate(masks):
        output_path = class_folder_path / f"mask_{image_name}_{mask_id}.png"
        PILImage.fromarray(mask).save(output_path)


if __name__ == "__main__":
    fire.Fire(test_hough_transform_on_dataset)
