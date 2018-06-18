from os.path import join, exists, split
from os import makedirs
import argparse
import numpy as np
from scipy.signal import convolve2d
import cv2


image = cv2.imread("/Users/marc/Documents/Dataset/Aquisition Images Final/Polarcam/image_00001.png")
image = image[:,:,1]

cv2.namedWindow('h', cv2.WINDOW_NORMAL)
cv2.imshow('h', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
