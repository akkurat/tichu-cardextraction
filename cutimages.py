from os.path import join, isfile
from pathlib import Path

import os

import numpy as np
import cv2
import re

def cutRect(img, rx0, rx1, ry0, ry1):

    width = img.shape[1]
    height = img.shape[0]
    x0 = int(rx0 * width)
    x1 = int(rx1 * width)
    y0 = int(ry0 * height)
    y1 = int(ry1 * height)
    uleft = img[y0:y1, x0:x1, :]
    return uleft

path = './cards/'
for f in os.listdir(path):
    full_path = join(path, f)
    if not isfile(full_path) or f.startswith('.'):
        continue

    img = np.zeros((10, 10))

    # img = cv2.imread(full_path)
    img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

    # # Only Corner for keypoints + rounded edge excluded
    # for iy, ix in np.ndindex(mask.shape):
    #     if ix < 0.19 * width and iy < 0.3 * height or ix > 0.81 * width and iy > 0.7 * height:
    #         if ix + iy > 20 and (width - ix) + (height - iy) > 20 and ix + height - iy > 20 and width - ix + iy > 20:
    #             mask[iy, ix] = 255

    uleftRank = cutRect(img, 0.02, 0.17, 0.04, 0.18)
    uleftSuit = cutRect(img, 0.02, 0.17, 0.18, 0.3)

    cv2.imshow('uleft', uleftRank)

    search = re.search('([a-z])(\d+|[A-Z]+)', f)
    if search:
        suit = search.group(1)
        rank = search.group(2)
        rpath = f'./classif/ranks/{rank}/{suit}.png'
        Path(rpath).parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(rpath, uleftRank)
        spath = f'./classif/suits/{suit}/{rank}.png'
        Path(spath).parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(spath, uleftSuit)


        # cv2.waitKey()

