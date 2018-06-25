from PIL import Image
import random
from random import randint
import numpy as np
from scipy.signal import convolve2d
import cv2
import colorsys


def rotateStandard(rotation, im):
    try:
        im = Image.fromarray(np.array(im))
    except e:
        pass

    im = im.rotate(rotation)

    im = np.array(im)

    return im


def rotatePola(rot, im):
    rotation = rot

    im = np.array(im)

    imageH = im[:, :, 0]
    tmp = np.array(imageH + (2 * rot))
    tmp = np.mod(tmp, 360)
    imageH = tmp

    rotated = im
    rotated[:, :, 0] = imageH

    im = Image.fromarray(np.array(rotated))

    im = im.rotate(rotation)
    im = np.array(im)

    return im
