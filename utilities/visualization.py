from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image as PILImage
from PIL import ImageDraw
from skimage import img_as_ubyte

from development.hough_transform.dev_hough_transform import Image


def insert_circles_in_image(
    image: Image,
    centroid_x: ArrayLike,
    centroid_y: ArrayLike,
    radii: ArrayLike,
    color: Optional[Tuple] = None,
    line_width: int = 3,
) -> Image:
    """Inserts a list of circles into an image.

    :param image: input image
    :param centroid_x: array of centroid x-coordinates
    :param centroid_y: array of centroid y-coordinates
    :param radii: array of radii
    :param color: color to use for the drawing of the circles, optional
    :param line_width: line width to use for the drawing of the circles
    :return: image with inserted circles
    """

    if color is None:
        color = (255, 0, 0)

    image = PILImage.fromarray(img_as_ubyte(image))

    image_drawer = ImageDraw.Draw(image)

    for center_y, center_x, radius in zip(centroid_y, centroid_x, radii):
        bounding_box = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        image_drawer.ellipse(bounding_box, outline=color, width=line_width)

    return np.array(image)
