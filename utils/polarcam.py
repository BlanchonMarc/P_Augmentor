# -*- coding: utf-8 -*-
"""
Created on Thu May 24 16:20:00 2018

@author: olivierm

Tools to read and import data from Polarcam
"""


import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


# -- Path of the flatfield file
FLATF = np.genfromtxt('/Users/marc/Github/P_Augmentor/utils/flatfield.csv',
                      dtype=float, delimiter=',', skip_header=9)

# %% Pixel order

# defined clockwise
# +-------+
# | 0 | 1 |
# |---+---|
# | 3 | 2 |
# +-------+

# by default
# +----------+
# | I0 | I45 |
# |----+-----|
# |I135| I90 |
# +----------+

#---------------I0-I45-I90-I135
I0_45_90_135 = (0, 1, 2, 3)
I45_0_135_90 = (1, 0, 3, 2)
I135_90_45_0 = (3, 2, 1, 0)
I90_135_0_45 = (2, 3, 0, 1)
I45_90_135_0 = (3, 0, 1, 2)

PIXELS_ORDER = I45_90_135_0

# %% Functions

def raw2quad(raw, method='none'):
    r"""Convert a raw image from polarcam into a list of ordered images
    [I0, I45, I90, I135]. If the parameter `method` is set to none the output images will have
    half size of the original image.

    Parameters
    ----------
    raw : 2D array
        Original polarcam image in grayscale
    method : {'none', 'linear', 'bilinear', 'weightedb3', 'weightedb4'}
        'none' : no interpolation method is performed
        'linear' : linear interpolation
        'bilinear' : bilinear interpolation
        'weightedb3' or 'weightedb4' : weighted bilinear interpolation with 3
        or 4 neighbors

    Returns
    -------
    images : 3D array containing 4 images

    Example
    -------
    >>> images = raw2quad(raw, method='bilinear')

    """

    if method == 'none':
        images = np.array([raw[0::2, 0::2],  # 0
                           raw[0::2, 1::2],  # 1
                           raw[1::2, 1::2],   # 2
                           raw[1::2, 0::2]])  # 3
        return images[PIXELS_ORDER, :, :]
    if method == 'linear':
        kernels = [np.array([[1, 0], [0, 0.]]),
                   np.array([[0, 1], [0, 0.]]),
                   np.array([[0, 0], [0, 1.]]),
                   np.array([[0, 0], [1, 0.]])]
    elif method == 'bilinear':
        kernels = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0.]]),
                   np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]) / 2.,
                   np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]) / 4.,
                   np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]) / 2.]
    elif method == 'weightedb3':
        b = np.sqrt(2) / 2 / (np.sqrt(2) / 2 + np.sqrt(10))
        a = np.sqrt(10) / 2 / (np.sqrt(2) / 2 + np.sqrt(10))

        kernels = [np.array([[0, b, 0, 0],
                             [0, 0, 0, 0],
                             [0, a, 0, b],
                             [0, 0, 0, 0]]),
                   np.array([[0, 0, b, 0],
                             [0, 0, 0, 0],
                             [b, 0, a, 0],
                             [0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0],
                             [b, 0, a, 0],
                             [0, 0, 0, 0],
                             [0, 0, b, 0]]),
                   np.array([[0, 0, 0, 0],
                             [0, a, 0, b],
                             [0, 0, 0, 0],
                             [0, b, 0, 0]])]
    elif method == 'weightedb4':
        c = np.sqrt(2)/2/(3*np.sqrt(2)/2+np.sqrt(2)/2+np.sqrt(10))
        b = np.sqrt(10)/2/(3*np.sqrt(2)/2+np.sqrt(2)/2+np.sqrt(10))
        a = 3*np.sqrt(2)/2/(3*np.sqrt(2)/2+np.sqrt(2)/2+np.sqrt(10))

        kernels = [np.array([[0, b, 0, c],
                             [0, 0, 0, 0],
                             [0, a, 0, b],
                             [0, 0, 0, 0]]),
                   np.array([[c, 0, b, 0],
                             [0, 0, 0, 0],
                             [b, 0, a, 0],
                             [0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0],
                             [b, 0, a, 0],
                             [0, 0, 0, 0],
                             [c, 0, b, 0]]),
                   np.array([[0, 0, 0, 0],
                             [0, a, 0, b],
                             [0, 0, 0, 0],
                             [0, b, 0, c]])]

    Is = [convolve2d(raw, k, mode='same') for k in kernels]
    offsets = [[(0, 0), (0, 1), (1, 1), (1, 0)],
               [(0, 1), (0, 0), (1, 0), (1, 1)],
               [(1, 1), (1, 0), (0, 0), (0, 1)],
               [(1, 0), (1, 1), (0, 1), (0, 0)]]

    images = np.zeros((4,)+raw.shape)
    for (j, o) in enumerate(offsets):
        for ide in range(4):
            images[j, o[ide][0]::2, o[ide][1]::2] = Is[ide][o[ide][0]::2, o[ide][1]::2]

    return images[PIXELS_ORDER, :, :]


def quad2pola(images, *args):
    r""" Convert a set of 4 images into polarization parameters

    Parameters
    ----------
    images : 3D array
        Containing the 4 images I0, I45, I90, I135
    *args : 'int', 'aop', 'dop', 'err'
        Variable parameters describing the key of the output dictionnary.

    Returns
    -------
    dic : dictionnary
        Specified parameters in *args are return into a dictionary

    Examples
    --------
    >>> quad2pola(images)  # return all polarization parameters in a dictionnary

    >>> quad2pola(images, 'aop', 'dop')  # return aop and dop parameters in a dictionnary
    """
     # --- Stokes parameters
    mat = np.array([[0.5, 0.5, 0.5, 0.5],
                    [1.0, 0.0, -1., 0.0],
                    [0.0, 1.0, 0.0, -1.]])

    stokes = np.tensordot(mat, images, 1)

    # --- Other parameters
    compl = stokes[1]+stokes[2]*1j
    aop = np.angle(compl) / 2.
    dop = np.divide(abs(compl), stokes[0], out=np.zeros_like(stokes[0]), where=stokes[0] != 0)


    imat = 0.5 * np.array([[1, 1, 0.],
                           [1, 0, 1.],
                           [1, -1, 0],
                           [1, 0, -1]])

    error = sum((images - np.tensordot(np.dot(imat, mat), images, 1))**2, 0)

    dico = {'aop':aop, 'dop':dop, 'int': stokes[0], 'err':error}

    if not args:
        return dico

    return {k:dico[k] for k in filter(dico.has_key, args)}


def pola2rgb(pola):
    """def pola2rgb dic to RGB """
        # --- RGB image
#    aop[satur] = np.nan
#    aop[dop > 1] = np.nan
#    aop[dop <= 0.01] = np.nan
#    cmap = plt.get_cmap('prism')
#    cmap.set_bad((0, 0, 0))
#    cmap.set_under((0, 0, 0))
#    aop_RGB = cmap(np.mod(aop, np.pi)/np.pi)
#    aop_RGB = np.uint8(aop_RGB[:, :, [2, 1, 0]]*255)  # opencv compatibilty
    pass


# %% Read Polar

def polaread(*args, **kwargs):
    r""" Read a raw image and return polarization parameters

    Parameters
    ----------
    filename : string
        file path
    method : {'none', 'linear', 'bilinear', 'weightedb3', 'weightedb4'}
        'none' : no interpolation method is performed
        'linear' : linear interpolation
        'bilinear' : bilinear interpolation
        'weightedb3' or 'weightedb4' : weighted bilinear interpolation with 3
        or 4 neighbors
    *args :  'int', 'aop', 'dop', 'err'
        Variable parameters describing the key of the output dictionnary.

    Return
    ------
    dic : dictionnary
        Specified parameters in *args are return into a dictionary

    Examples
    --------
    >>> polaread('filename.tiff')  # open file and return all polarization parameters in a dictionnary

    >>> polaread('filename.tiff', 'aop', 'dop')  # open file and return aop and dop parameters in a dictionnary
    """

    method = kwargs.pop('method', 'none')
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    filename = args[0]

    # --- Open the image
    im_raw = plt.imread(filename)

    # --- Saturated pixels
    #satur = reduce(np.bitwise_and, raw) == 255  # only valid with U8 images

    im_flat = im_raw * FLATF
    # --- Extract superpixels
    images = raw2quad(im_flat, method)

    dico = quad2pola(images, *args[1:])

    dico['raw'] = im_flat

    return dico

#
# # %% Main
#
# if __name__ == '__main__':
#     # Direct call of the script
#     TEST = polaread('../progs-spyder/Datasets/2017-03-14-Rz/frame-0000.tiff', method='linear')
#
#
#     from synthetic import create_synth
#     TEST = create_synth()
#     raw = TEST['raw']
#
#     superpixel = quad2pola(raw2quad(raw))['aop']
#     linear = quad2pola(raw2quad(raw, method='linear'))['aop']
#     bilin = quad2pola(raw2quad(raw, method='bilinear'))['aop']
#     w3 = quad2pola(raw2quad(raw, method='weightedb3'))['aop']
#
#     plt.figure()
#     plt.subplot(1, 3, 1)
#     plt.imshow(superpixel, cmap='jet')
#     plt.subplot(1, 3, 2)
#     plt.imshow(linear, cmap='jet')
#     plt.subplot(1, 3, 3)
#     plt.imshow(bilin, cmap='jet')
#
