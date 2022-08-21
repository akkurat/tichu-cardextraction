import math
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from fontTools.svgLib.path.arc import TWO_PI

from help import refCornerHL, refCornerLR, findHull, methods

sift = cv2.SIFT_create()

debug = True
cards = []
for f in glob('./cards/*'):
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hullHL = findHull(img, refCornerHL, debug=debug)
    hullLR = findHull(img, refCornerLR, debug=debug)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask = cv2.fillPoly(mask, pts=[hullHL], color=255)
    kp, desc = sift.detectAndCompute(gray, mask)
    cards.append({"name": f, "img": img, "kp": kp, "desc": desc, "mask": mask})

    if debug:
        img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('card', img)
        # cv2.imshow('mask', mask)


cardnames = list(map(lambda c: c['name'], cards))


plt.ion()
fig = plt.figure()

cap = cv2.VideoCapture(0)
key = None
while key != 27:
    _, frame = cap.read()
    # frame = cv2.imread('./manycards2.jpg')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(gray, None)

    img = cv2.drawKeypoints(gray, kp, frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('keypoints', img)
    # cv2.imshow('frame', frame)
    # blur = cv2.bilateralFilter(frame, 20, 100, 500)
    # cv2.imshow('blur', blur)

    # plt.imshow(frame), plt.show()

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    results = []


    for c in cards:
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc, c['desc'], k=2)
        # sorted(matches, key=lambda x: x.distance)
        matches = [m for m, n in matches if m.distance < 0.65 * n.distance]

        results.append(len(matches))

        imgMatches = cv2.drawMatches(img, kp, c['img'], c['kp'], matches[:50], img, flags=2)

        matchBack = [kp[m.queryIdx] for m in matches]
        matchFrom = [c['kp'][m.trainIdx] for m in matches]

        # cv2.imshow(c['name'], imgMatches)
        backCircles = np.zeros(frame.shape, np.uint8)
        imgRelLines = np.zeros(frame.shape, np.uint8)
        fromCircles = np.zeros(c['img'].shape, np.uint8)
        for mb in matchBack:
            center = (int(mb.pt[0]), int(mb.pt[1]))
            cv2.circle(backCircles, center, 10, (255, 255, 255), 5)

        for mf in matchFrom:
            center = (int(mf.pt[0]), int(mf.pt[1]))
            cv2.circle(fromCircles, center, 10, (255, 255, 255), 5)

        for (mb,mf) in zip(matchBack, matchFrom):
            center = (int(mb.pt[0]), int(mb.pt[1]))
            relAngle = (mb.angle-mf.angle)/TWO_PI
            direction = (center[0]+int(math.cos(relAngle)*100), center[1]+int(math.sin(relAngle)*100))
            cv2.line(imgRelLines, center, direction, (255,255,255), 5)




        backName = 'cam' + c['name']
        fromName = 'card' + c['name']
        # cv2.imshow(backName, backCircles)
        # cv2.imshow(fromName, fromCircles)
        cv2.imshow(backName, imgRelLines)

    plt.clf()
    plt.barh(cardnames, results)
    plt.pause(0.01)

    print(results)

    if debug:
        key = cv2.waitKey(1)

cv2.destroyAllWindows()
cap.release()
