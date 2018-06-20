from os.path import join, exists, split
from os import makedirs
import argparse
import numpy as np
from scipy.signal import convolve2d
import cv2
import colorsys
from PIL import Image

image = cv2.imread("/Users/marc/Downloads/2.tiff")
image = image[:, :, 0]

raw = image
quad = np.vstack((np.hstack((image[0::2, 0::2],
                            image[0::2, 1::2])),
                  np.hstack((image[1::2, 1::2],
                            image[1::2, 0::2]))))
images = []

kernels = [np.array([[1, 0], [0, 0.]]),
           np.array([[0, 1], [0, 0.]]),
           np.array([[0, 0], [0, 1.]]),
           np.array([[0, 0], [1, 0.]])]

Is = []
for k in kernels:
    Is.append(convolve2d(raw, k, mode='same'))

offsets = [[(0, 0), (0, 1), (1, 1), (1, 0)],
           [(0, 1), (0, 0), (1, 0), (1, 1)],
           [(1, 1), (1, 0), (0, 0), (0, 1)],
           [(1, 0), (1, 1), (0, 1), (0, 0)]]

images = []
for (j, o) in enumerate(offsets):
    images.append(np.zeros(raw.shape))
    for ide in range(4):
        images[j][o[ide][0]::2, o[ide][1]::2] = Is[ide][o[ide][0]::2,
                                                        o[ide][1]::2]

Js = images
inten = (Js[0] + Js[1] + Js[2] + Js[3]) / 2.
aop = (0.5 * np.arctan2(Js[1] - Js[3], Js[0] - Js[2]))
dop = np.sqrt((Js[1] - Js[3])**2 + (Js[0] - Js[2])**2) / (Js[0] + Js[1] + Js[2] + Js[3] + np.finfo(float).eps) * 2

hsv = np.uint8(cv2.merge(((aop + np.pi / 2) / np.pi * 180,
                          dop * 255,
                          inten / inten.max() * 255)))

"""
Transform Translate Vertical top
"""
tmpprev = np.array(hsv)
translation = 100
translated = np.array(hsv)
translated[translation:, :, :] = translated[:-translation:, :, :]
translated[:translation, :, :] = 0
translated = cv2.cvtColor(translated, cv2.COLOR_HSV2RGB)
translated = Image.fromarray(np.array(translated))
# translated.show()

"""
Transform Translate Vertical bottom
"""
tmpprev = np.array(hsv)
translation = tmpprev.shape[2] - 100
translated = np.array(hsv)
translated[:translation, :, :] = translated[-translation:, :, :]
translated[translation:, :, :] = 0
translated = cv2.cvtColor(translated, cv2.COLOR_HSV2RGB)
translated = Image.fromarray(np.array(translated))
# translated.show()

"""
Transform Translate Horizontal left
"""
tmpprev = np.array(hsv)
translation = 100
translated = np.array(hsv)
translated[:, translation:, :] = translated[:, :-translation, :]
translated[:, :translation, :] = 0
translated = cv2.cvtColor(translated, cv2.COLOR_HSV2RGB)
translated = Image.fromarray(np.array(translated))
# translated.show()

"""
Transform Translate Horizontal right
"""
tmpprev = np.array(hsv)
translation = tmpprev.shape[1] - 100
translated = np.array(hsv)
translated[:, :translation, :] = translated[:, -translation:, :]
translated[:, translation:, :] = 0
translated = cv2.cvtColor(translated, cv2.COLOR_HSV2RGB)
translated = Image.fromarray(np.array(translated))
translated.show()

"""
Transform Rotate
"""

rotation = 0
rotation = rotation % 360

DegToH = np.linspace(0, 360, num=360)

imageH = hsv[:, :, 0]
tmp = np.array(imageH + (2 * rotation))
tmp = np.mod(tmp, 360)
imageH = tmp

hsv[:, :, 0] = imageH


rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

rgb = Image.fromarray(np.array(rgb))

rgb = rgb.rotate(rotation)

# rgb.show()
