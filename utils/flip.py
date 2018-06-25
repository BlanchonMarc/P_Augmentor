from PIL import Image
import random
from random import randint
import numpy as np
from scipy.signal import convolve2d
import cv2
import colorsys


def flipXpola(im):
    tmpprev = np.array(im)

    imageH = tmpprev[:, :, 0]
    tmp = np.array(-1 * imageH + 360)
    tmp = np.mod(tmp, 360)
    imageH = tmp

    tmpprev[:, :, 0] = imageH

    im = Image.fromarray(np.array(tmpprev))

    im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = np.array(im)

    return im


def flipYpola(im):
    tmpprev = np.array(im)

    imageH = tmpprev[:, :, 0]
    tmp = np.array(-1 * imageH + 360)
    tmp = np.mod(tmp, 360)
    imageH = tmp

    tmpprev[:, :, 0] = imageH

    im = Image.fromarray(np.array(tmpprev))

    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im = np.array(im)

    return im


def flipXstandard(im):
    try:
        im = Image.fromarray(np.array(im))
    except e:
        pass

    im = im.transpose(Image.FLIP_LEFT_RIGHT)

    return im


def flipYstandard(im):
    try:
        im = Image.fromarray(np.array(im))
    except e:
        pass

    im = im.transpose(Image.FLIP_TOP_BOTTOM)

    return im
