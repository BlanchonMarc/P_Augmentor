import yaml
import glob
import os
import sys
from PIL import Image
import random
from random import randint
import numpy as np
from scipy.signal import convolve2d
import cv2
import colorsys
from utils import translate
from utils import rotate
from utils import flip

"""
Open the YAML
"""

with open("example.yaml", 'r') as stream:
    try:
        data = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# print(data)

"""
Parse the YAML
"""

files = [[0, 0], [0, 0]]

for indx in range(len(data[0]['inputs']['paths'])):

    os.chdir(data[0]['inputs']['paths'][indx])
    for file in glob.glob("*." + data[0]['inputs']['extansions'][indx]):
        files[indx].append(file)


for indx in range(0, 2):
    for _ in range(0, 2):
        del files[indx][0]

im_type = []

for types in data[0]['inputs']['type']:
    im_type.append(types)

trans_type = []

for types in data[1]['constraints']['type']:
    trans_type.append(types)

increments = []

for increment in data[1]['constraints']['increment']:
    increments.append(increment)

aug_per_im = data[1]['constraints']['nb_im_out']

path_out = data[2]['output']['path']

"""
Building the vector of possible combinations with parsed informations
"""

iterrot = []
itertranX = []
itertranY = []
probX = 0
probY = 0

if len(increments) == len(trans_type):
    for main_indx in range(len(trans_type)):
        if trans_type[main_indx] == 'rotation':
            # print('rotation')
            for indx in range(len(increments)):
                iterrot = [increments[main_indx] * i
                           for i in range(int(360 / increments[main_indx]) + 1)]

        elif trans_type[main_indx] == 'translation':
            # print('translation')
            im = Image.open(files[0][0])
            width, height = im.size
            itertranX = [increments[main_indx] * i
                         for i in range(10)]

            tmp = [-x for x in itertranX]
            [tmp.append(x) for x in itertranX if not x == 0]
            itertranX = sorted(tmp)

            itertranY = [increments[main_indx] * i
                         for i in range(10)]

            tmp = [-x for x in itertranY]
            [tmp.append(x) for x in itertranY if not x == 0]
            itertranY = sorted(tmp)
        elif trans_type[main_indx] == 'flipX':
            probX = increments[main_indx]
        elif trans_type[main_indx] == 'flipY':
            probY = increments[main_indx]
        else:
            raise NameError("Unknown type of contraints")
else:
    raise NameError("contraints: increments and types have inequal length")

"""
Creating the combinations
"""

combinations = [[0, 0, 0, 0, 0]]

for comb in range(aug_per_im - 1):
    temp_comb = []
    rot_comb = iterrot[randint(0, len(iterrot) - 1)]
    tranX_comb = itertranX[randint(0,
                                   len(itertranX) - 1)]
    tranY_comb = itertranY[randint(0,
                                   len(itertranY) - 1)]

    prob_test = random.random()
    flip_combX = 0
    if prob_test < probX:
        flip_combX = 1

    prob_test = random.random()
    flip_combY = 0
    if prob_test < probY:
        flip_combY = 1

    temp_comb.append(rot_comb)
    temp_comb.append(tranX_comb)
    temp_comb.append(tranY_comb)
    temp_comb.append(flip_combX)
    temp_comb.append(flip_combY)

    combinations.append(temp_comb)


nb_im = 1


for indFolder in range(len(files)):

    if im_type[indFolder] == 'polar':
        folderPola = files[indFolder]
        try:
            indexgt = im_type.index('gt')
        except e:
            print('Corresponging GT for pola is necessary')
            sys.exit(1)

        folderGT = files[indFolder]

        for index in range(len(folderPola)):
            """
            POLARIMETRIC AUGMENTATION
            """
            """
            Reading the image with opencv
            """

            image = Image.open(data[0]['inputs']['paths'][
                indFolder] + folderPola[index])

            image = np.array(image)

            """
            Converting the image into Interpolable HSL if Pola
            """
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
                    images[j][o[ide][0]::2, o[ide][1]::2] = Is[ide][
                        o[ide][0]::2, o[ide][1]::2]

            Js = images
            inten = (Js[0] + Js[1] + Js[2] + Js[3]) / 2.
            aop = (0.5 * np.arctan2(Js[1] - Js[3], Js[0] - Js[2]))
            dop = np.sqrt((Js[1] - Js[3])**2 + (
                Js[0] - Js[2])**2) / (
                    Js[0] + Js[1] + Js[2] + Js[3] + np.finfo(float).eps) * 2

            hsvinit = np.uint8(cv2.merge(((aop + np.pi / 2) / np.pi * 180,
                                          dop * 255,
                                          inten / inten.max() * 255)))

            """
            Now Augment
            """
            for aug in range(len(combinations)):
                hsv = hsvinit
                curr_aug = combinations[aug]

                if curr_aug[3] == 1:
                    hsv = flip.flipXpola(hsv)
                if curr_aug[4] == 1:
                    hsv = flip.flipYpola(hsv)

                hsv = rotate.rotatePola(curr_aug[0], hsv)

                hsv = translate.translateX(curr_aug[2], hsv)

                hsv = translate.translateY(curr_aug[1], hsv)

                path_polar = path_out + 'polar/'
                if not os.path.exists(path_out + 'polar/'):
                    os.makedirs(path_out + 'polar/')

                number = format(nb_im, '05')

                converted = cv2.cvtColor(np.array(hsv), cv2.COLOR_HSV2RGB)
                im_fin = Image.fromarray(np.array(converted))

                im_fin.save(path_polar + 'image_' + number + '.png')

                """
                RGB
                """

                rgb = Image.open(data[0]['inputs']['paths'][
                    indexgt] + folderGT[index])

                rgb = np.array(rgb)

                rgb = Image.fromarray(np.array(rgb))

                if curr_aug[3] == 1:
                    rgb = flip.flipXstandard(rgb)
                if curr_aug[4] == 1:
                    rgb = flip.flipYstandard(rgb)

                rgb = rotate.rotateStandard(curr_aug[0], rgb)

                rgb = translate.translateX(curr_aug[2], rgb)

                rgb = translate.translateY(curr_aug[1], rgb)

                path_gt = path_out + 'gt/'
                if not os.path.exists(path_gt):
                    os.makedirs(path_gt)

                number = format(nb_im, '05')

                im_fin = Image.fromarray(np.array(rgb))

                im_fin.save(path_gt + 'image_' + number + '.png')

                nb_im += 1
