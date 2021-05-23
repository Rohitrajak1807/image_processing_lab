import cv2

import question_2 as q
import numpy as np


def thr(i):
    if i > 128:
        return 255
    else:
        return 0


def m(x):
    a = [thr(i) for i in x]
    return a


img = q.read_image('../res/img.png', flags=cv2.IMREAD_GRAYSCALE)
img2 = np.asarray([m(x) for x in img], dtype=np.uint8)
q.display_image(img2)
print(img2)
