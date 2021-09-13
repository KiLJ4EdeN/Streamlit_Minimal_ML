# preprocessing module
import numpy as np
import cv2
from PIL import Image, ImageOps

reference_shape = (28, 28, 1)


def preprocess_image(image):
    """
    :param image: mnist image to be processed
    :return: preprocessed image ready for prediction
    """
    # pil preprocessing
    image = ImageOps.fit(image, (28, 28), Image.ANTIALIAS)
    image = np.asarray(image)
    # pil to cv
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # check channel
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compatible size
        if image.shape != reference_shape:
            image = cv2.resize(image, (28, 28)) / 255.
    return image

