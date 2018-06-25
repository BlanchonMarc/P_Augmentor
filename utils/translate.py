from PIL import Image
import random
from random import randint
import numpy as np
from scipy.signal import convolve2d
import cv2
import colorsys


def translateX(val, im):
    if val == 0 or val == im.shape[0]:
        return im
    else:
        if val < 0:
            tmpprev = np.array(im)
            translation = tmpprev.shape[0] - (val * -1)
            translated = np.array(im)
            translated[:translation, :] = translated[-translation:, :]
            translated[translation:, :] = 0
            im = translated

        else:

            translation = val
            translated = np.array(im)
            translated[translation:, :] = translated[:-translation, :]
            translated[:translation, :] = 0
            im = translated

        return im


def translateY(val, im):
    if val == 0 or val == im.shape[1]:
        return im
    else:
        if val < 0:

            tmpprev = np.array(im)
            translation = tmpprev.shape[1] - (val * -1)
            translated = np.array(im)
            translated[:, :translation] = translated[:, -translation:]
            translated[:, translation:] = 0
            im = translated

        else:
            translation = val
            translated = np.array(im)
            translated[:, translation:] = translated[:, :-translation]
            translated[:, :translation] = 0
            im = translated

        return im
