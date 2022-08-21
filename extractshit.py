# Test find_hull on a random card image
# debug = "no" or "pause_always" or "pause_on_pb"
# If debug!="no", you may have to press a key to continue execution after pause
import random
from glob import glob

import cv2

from displayimage import display_img
from help import findHull, refCornerHL, refCornerLR

debug="yes"
imgs_dir="./cards"
imgs_fns=glob(imgs_dir+"/*.png")
img_fn=random.choice(imgs_fns)
print(img_fn)
img=cv2.imread(img_fn,cv2.IMREAD_UNCHANGED)

hullHL=findHull(img,refCornerHL,debug=debug)
hullLR=findHull(img,refCornerLR,debug=debug)
display_img(img,[refCornerHL,refCornerLR,hullHL,hullLR])

cv2.waitKey(0)

if debug!="no": cv2.destroyAllWindows()