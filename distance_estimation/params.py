from random import seed
from random import randint
import numpy as np

seed(1)

def get_bounding_box():
    x = randint(0, 540)
    y = randint(180, 480)
    x2 = x + 100
    y2 = y - 180;
    return [x, y, x2, y2]

def bounding_boxes():
    boxes = []
    for i in range(0, 10):
        boxes.append(get_bounding_box())
    return boxes

def frame(boxes):
    img = np.zeros((640, 480, 1))
    for box in boxes:
        x,y,x2,y2 = box[0], box[1], box[2], box[3]
        img[x:x2, y] = 1
        img[x:x2, y2] = 1
        img[x, y2:y] = 1
        img[x2, y2:y] = 1
    return img