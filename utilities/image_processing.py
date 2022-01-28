from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from skimage import color
from skimage.filters import threshold_otsu

Image = ArrayLike
Mask = ArrayLike


def apply_otsu_threshold(image: Image, mask: Optional[Mask] = None) -> Mask:
    """Creates a binary image using Otsus method.

    :param image: input image
    :param mask: mask to limit the calculation of the threshold to a certain area
    :return: binary image
    """

    if mask is None:
        mask = np.ones(image.shape, dtype=bool)

    image = rgb2gray(image)

    threshold = threshold_otsu(image[mask])
    return np.greater(image, threshold)


def rgb2gray(image: Image) -> Image:
    """Convert image to grayscale, if necessary.

    :param image: input image (potentially rgb)
    :return: grayscale image
    """
    if image.ndim == 3:
        image = color.rgb2gray(image)
    return image


def apply_mean_threshold(image: Image, mask: Mask) -> Mask:
    """Performs an image binarization, using the mean value of the image as threshold.

    :param image: input image
    :param mask: mask to limit the calculation of the threshold to a certain area
    :return: binary image
    """
    image = rgb2gray(image)

    binary_image = image > np.mean(image[mask])
    return binary_image
