from random import seed
from random import randint
import numpy as np
from skimage import io
seed(1)

def get_bounding_box():
    x = randint(0, 540)
    y = randint(180, 480)
    x2 = x + 100
    y2 = y - 180;
    return [x, y, x2, y2]

def bounding_boxes():
    # boxes = []
    # for i in range(0, 10):
    #     boxes.append(get_bounding_box())
    return [
        [627,549,743,812],
        [886,750,982,1000],
        [902,161,957,316],
        [967,177,1012,332],
    ]

def frame(boxes):
    img = io.imread("./test.png")

    for box in boxes:
        x,y,x2,y2 = box
        img[y,x:x2] = [0,0,0]
        img[y2, x:x2] = [0,0,0]
        img[y:y2, x] = [0,0,0]
        img[y:y2, x2] = [0,0,0]
    return img