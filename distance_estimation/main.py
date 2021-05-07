import cv2, numpy as np
from params import bounding_boxes, frame
from PIL import Image
from matplotlib import pyplot as plt

# Should be taken from sensors or reference points in our system...
calibration = [[180,162],[618,0],[552,540],[682,464]]


def image_to_bird(img, calibration):
    img_width, img_height = img.shape[0:2]
    src_point = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
    dest_point = np.float32(calibration)
    transform_matrix = cv2.getPerspectiveTransform(src_point, dest_point)
    return cv2.warpPerspective(img, transform_matrix, (img_width, img_height))

def bird_to_image(img, calibration):
    img_width, img_height = img.shape[0:2]
    src_point = np.float32(calibration)
    dest_point = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]]) 
    transform_matrix = cv2.getPerspectiveTransform(src_point, dest_point)
    return cv2.warpPerspective(img, transform_matrix, (img_width, img_height))


def project_point_on_bird(img, calibration, p):
    img_width, img_height = img.shape[0:2]
    src_point = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
    dest_point = np.float32(calibration)
    transform_matrix = cv2.getPerspectiveTransform(src_point, dest_point)
    M = transform_matrix
    px = (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
    py = (M[1][0] * p[0] + M[1][1] * p[1] + M[1][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
    return (int(px), int(py))

def project_point_on_image(img, calibration, p):
    img_width, img_height = img.shape[0:2]
    src_point = np.float32(calibration)
    dest_point = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]]) 
    transform_matrix = cv2.getPerspectiveTransform(src_point, dest_point)
    M = transform_matrix
    px = (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
    py = (M[1][0] * p[0] + M[1][1] * p[1] + M[1][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
    return (int(px), int(py))


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def centroids(boxes, image, calibration):
    bird_view = image_to_bird(image.copy(), calibration)
    centroids = []
    for b in boxes:
        xmin, ymin, xmax, ymax = b[0], b[1], b[2], b[3]
        w = xmax - xmin
        h = ymax - ymin
        x = xmin + w/2
        y = ymax - h/2
        bird_x, bird_y = project_point_on_bird(bird_view, calibration, (x, ymax))
        center_bird_x, center_bird_y = project_point_on_bird(bird_view, calibration, (x, ymin))

        centroids.append((
            int(bird_x), int(bird_y),
            int(x), int(ymax), 
            int(xmin), int(ymin), int(xmax), int(ymax),
            int(center_bird_x), int(center_bird_y)
        ))

    return centroids

boxes = bounding_boxes()
_frame = frame(boxes)

fig_1 = plt.figure("Bounding boxes")
plt.imshow(_frame, interpolation='nearest')

_frame_bird = image_to_bird(_frame, calibration)

fig_2 = plt.figure("Bird Eye view")
plt.imshow(_frame_bird, interpolation='nearest')

## Box -> x1,y1,x2,y2
_centroids = centroids(boxes, _frame, calibration)

distances = []

for i in range(0, len(_centroids)):
    ith_distances = []
    for j in range(0, len(_centroids)):
        ith_distances.append(distance(_centroids[i], _centroids[j]))
    distances.append(ith_distances)

fig_3 = plt.figure("Distance Heat map")
c = plt.imshow(distances, interpolation='nearest', cmap="RdBu")
plt.colorbar(c)
plt.show()

