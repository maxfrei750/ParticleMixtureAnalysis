from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from skimage import draw
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks

from utilities.image_processing import rgb2gray

Image = ArrayLike
Mask = ArrayLike
CircleDetectionResult = Tuple[ArrayLike, ArrayLike, ArrayLike]


def circle_detection_hough(
    image: Image,
    radius_min: int,
    radius_max: int,
    mask: Optional[Mask] = None,
    sigma_canny: float = 3.0,
) -> CircleDetectionResult:
    """Detects circular object in images using the Hough Transform and Canny edge detection.

    based on:
        https://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html

    :param image: input image
    :param radius_min: minimum radius to be detected
    :param radius_max: maximum radius to be detected
    :param mask: mask to confine the detection, optional
    :param sigma_canny: standard deviation of the Gaussian filter applied during the Canny edge
        detection, optional
    :return: tuple of arrays containing x-coordinates of centroids, y-coordinates of centroids and
        radii of the detected circles
    """

    image = rgb2gray(image)

    if mask is None:
        mask = np.full_like(image, True, dtype=bool)

    edges = canny(image, sigma=sigma_canny, mask=mask)

    # from skimage.morphology import binary_dilation
    #
    # plt.imsave(
    #     VISUALIZATION_ROOT / "edges_dark.png",
    #     binary_dilation(edges),
    #     cmap="gray",
    # )

    hough_radii = np.arange(radius_min, radius_max)
    hough_res = hough_circle(edges, hough_radii)

    min_distance = round((radius_min + radius_max) / 2)

    # `normalize=True` gave worse results
    accums, centroids_x, centroids_y, radii = hough_circle_peaks(
        hough_res,
        hough_radii,
        min_xdistance=min_distance,
        min_ydistance=min_distance,
    )

    centroids_x, centroids_y, radii = _filter_circle_detections_based_on_mask(
        centroids_x, centroids_y, radii, mask
    )

    return centroids_x, centroids_y, radii


def _filter_circle_detections_based_on_mask(
    centroids_x: ArrayLike, centroids_y: ArrayLike, radii: ArrayLike, mask: Mask
) -> CircleDetectionResult:
    """Filter circle detections based on a mask. Only detections that overlap more than 50% with the
        mask are kept.

    :param centroids_x: x-coordinates of circle centroids
    :param centroids_y: y-coordinates of circle centroids
    :param radii: radii of the circles
    :param mask: mask used for the filtering
    :return: tuple of filtered x-coordinates, y-coordinates and radii
    """

    if not np.any(mask):
        return np.array([]), np.array([]), np.array([])

    if not np.all(mask):
        is_relevant = []

        for centroid_x, centroid_y, radius in zip(centroids_x, centroids_y, radii):
            detection_area = np.pi * radius ** 2

            detection_mask_indices = draw.disk((centroid_y, centroid_x), radius, shape=mask.shape)

            interesting_area = np.sum(mask[detection_mask_indices])

            if interesting_area / detection_area > 0.5:
                is_relevant.append(True)
            else:
                is_relevant.append(False)

        centroids_x = centroids_x[is_relevant]
        centroids_y = centroids_y[is_relevant]
        radii = radii[is_relevant]

    return centroids_x, centroids_y, radii


def circle_detections_to_masks(
    image: Image, centroids_x: ArrayLike, centroids_y: ArrayLike, radii: ArrayLike
) -> Image:
    """Converts circle detections into masks of circles.

    :param image: input image
    :param centroids_x: x-coordinates of circle centroids
    :param centroids_y: y-coordinates of circle centroids
    :param radii: radii of the circles
    :return: array of masks
    """
    masks = []

    for centroid_y, centroid_x, radius in zip(centroids_y, centroids_x, radii):
        mask = np.full_like(image, False, dtype=bool)
        instance_indices = draw.disk((centroid_y, centroid_x), radius, shape=image.shape)
        mask[instance_indices] = True
        masks.append(mask)

    return np.asarray(masks)
