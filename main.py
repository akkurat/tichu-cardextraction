import math
import os
from os.path import isfile, join, isdir

import cv2
import numpy as np
from fontTools.svgLib.path.arc import TWO_PI
from sklearn.cluster import DBSCAN, MeanShift

threshKp = 0.3
RING = 8

matcher = cv2.SIFT_create()

debug = False
# debug = True
ranks = []
opath = './classif/ranks/'
for r in os.listdir(opath):
    path = join(opath, r)
    if not isdir(path):
        continue
    kp = ()
    fp = ()
    desc = None
    for f in os.listdir(path):
        full_path = join(path, f)
        if not isfile(full_path) or f.startswith('.'):
            continue
        img = cv2.imread(full_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # hullHL = findHull(img, refCornerHL, debug=debug)
        # hullLR = findHull(img, refCornerLR, debug=debug)
        width = img.shape[1]
        height = img.shape[0]

        # _fp = fast.detect(gray, None)
        _kp, _desc = matcher.detectAndCompute(gray, None)
        kp += _kp
        if desc is not None:
            if _desc is not None:
                desc = np.concatenate((desc, _desc))
        else:
            desc = _desc

    ranks.append({"name": r, "img": img, "kp": kp, "desc": desc})

    if debug:
        img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow(r + 'card', img)

if debug: cv2.waitKey()
cv2.destroyAllWindows()

## exclude too similar features

search_params = dict(checks=100)
index_params = dict(algorithm=0, trees=5)
flann = cv2.FlannBasedMatcher(index_params, search_params)

cardnames = list(map(lambda c: c['name'], ranks))


def ransac_points(src_pts, dst_pts):
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    return M, mask
    # h, w = img1.shape
    # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # dst = cv.perspectiveTransform(pts, M)
    # img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)


class CardDetector:

    def __init__(self):
        self.meanshift = MeanShift(cluster_all=False, bandwidth=75)
        self.moving = []
        self.i = 0

    # todo: param mask & imt2match
    def find_cards(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width = gray.shape[1]
        height = gray.shape[0]

        warped = np.zeros(frame.shape, dtype=frame.dtype)

        gb = int(width / 20)
        if gb % 2 == 0:
            gb += 1
        kernel = np.ones((gb // 2 + 1, gb // 2 + 1), np.uint8)
        thresh = cv2.GaussianBlur(gray, (gb, gb), 0)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        T, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        descriptors = matcher.detectAndCompute(gray, thresh)
        if (len(self.moving) < RING):
            self.moving.append(descriptors)
        else:
            self.moving[self.i] = descriptors

        self.i = (self.i + 1) % RING

        # ret, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imshow('thresh', thresh)

        kp = ()
        desc = None

        for _kp, _desc in self.moving:
            kp += _kp
            desc = np.concatenate((desc, _desc)) if desc is not None else _desc

        gray = cv2.drawKeypoints(gray, kp, frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('frame', frame)
        # blur = cv2.bilateralFilter(frame, 20, 100, 500)
        # cv2.imshow('blur', blur)
        # plt.imshow(frame), plt.show()
        results = {}
        for c in ranks:
            threshold = len(c['kp']) * threshKp
            _matches = flann.knnMatch(desc, c['desc'], k=2)
            # sorted(matches, key=lambda x: x.distance)
            # _matches = filter(lambda t: len(t) > 1, _matches)
            matches = [m for m, n in _matches if m.distance < 0.8 * n.distance]
            matchBack = [kp[m.queryIdx] for m in matches]
            matchFrom = [c['kp'][m.trainIdx] for m in matches]

            num_matches = len(matches)
            print( num_matches, threshold )
            if num_matches > threshold:

                results[c['name']] = ''

                src_pts = np.float32([kp.pt for kp in matchBack]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp.pt for kp in matchFrom]).reshape(-1, 1, 2)
                pts = np.float32([kp.pt for kp in matchBack])
                cluster = self.meanshift.fit(pts)
                # print(cluster.cluster_centers_)
                for i, center in enumerate(cluster.cluster_centers_):
                    int_center = center.astype(int)
                    color = ((i % 3) * 200, ((i + 1) % 3) * 200, ((i + 2) % 3) * 200)
                    cv2.circle(gray, int_center, 10, color, -1)
                    pt_mask = cluster.labels_ == i
                    cluster_src_pts = src_pts[pt_mask]
                    cluster_dst_pts = dst_pts[pt_mask]
                    # draw_boundingrect(color, cluster_src_pts, gray)

                    clustered_len = len(cluster_src_pts)
                    if (clustered_len > 10):
                        M, mask = ransac_points(cluster_src_pts, cluster_dst_pts)
                        biMask = mask > 0
                        filteredPoints = cluster_src_pts[biMask]
                        for p in filteredPoints:
                            cv2.circle(gray, np.int32(p), 5, (255, 36, 12), -1)

                        draw_boundingrect((255, 36, 12), filteredPoints, gray)
                        results[c['name']] += ' ' + str(len(filteredPoints))
                        if M is not None:
                            warped += cv2.warpPerspective(c['img'], M, (width, height))

            if debug:
                imgMatches = cv2.drawMatches(img, kp, c['img'], c['kp'], matches[:50], img, flags=2)

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

                for (mb, mf) in zip(matchBack, matchFrom):
                    center = (int(mb.pt[0]), int(mb.pt[1]))
                    relAngle = (mb.angle - mf.angle) / TWO_PI
                    direction = (center[0] + int(math.cos(relAngle) * 100), center[1] + int(math.sin(relAngle) * 100))
                    cv2.line(imgRelLines, center, direction, (255, 255, 255), 5)

                backName = 'cam' + c['name']
                fromName = 'card' + c['name']
                # cv2.imshow(backName, backCircles)
                # cv2.imshow(fromName, fromCircles)
                cv2.imshow(backName, imgRelLines)

        cv2.imshow('keypoints', gray)
        cv2.imshow('warp', warped)
        cv2.waitKey(1)
        print(results)


def draw_boundingrect(color, filteredPoints, img):
    x, y, w, h = cv2.boundingRect(filteredPoints)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)


img = cv2.imread('./curve.png')
# img = cv2.imread('./manycards2.jpg')
cm = CardDetector()
cm.find_cards(img)
cv2.waitKey()
exit(0)

cap = cv2.VideoCapture(0)
key = None
cm = CardDetector()
while key != 27:
    _, frame = cap.read()

    cm.find_cards(frame)

    if debug:
        key = cv2.waitKey(1)

cv2.destroyAllWindows()
cap.release()
