import shutil
from glob import glob

import cv2
import numpy as np

sift = cv2.SIFT_create()

debug = False
debug = True
cards = []

for f in glob('./refcards/*'):
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, desc = sift.detectAndCompute(gray, None)

    import os
    fname = os.path.basename(f)

    cards.append({"name": fname, "img": img, "kp": kp, "desc": desc})

    if debug:
        img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow(f + 'card', img)

if debug: cv2.waitKey()
cv2.destroyAllWindows()

index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=60)
flann = cv2.FlannBasedMatcher(index_params, search_params)

for f in glob('./out/*'):
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, desc = sift.detectAndCompute(gray, None)

    maxRating = 0; bestCard = None
    for c in cards:
        matches = flann.knnMatch(desc, c['desc'], k=2)
        goodones = [m for m, n in matches if m.distance < 0.2 * n.distance]
        rating = len(goodones)
        if rating > maxRating:
            maxRating = rating
            bestCard = c

    fname = './cards/' + bestCard['name']
    while os.path.exists(fname):
        fname += '-'

    shutil.copy2(f, fname)



