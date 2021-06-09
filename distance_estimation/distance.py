import cv2, numpy as np
from params import bounding_boxes, frame
from transformations import get_perspective_transform, warp_perspective
from PIL import Image
from matplotlib import pyplot as plt


def image_to_bird(img, calibration):
    img_width, img_height = img.shape[0:2]
    src_rect = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
    dest_rect = np.float32(calibration)
    transform_matrix = get_perspective_transform(src_rect, dest_rect)
    # cv2.warpPerspective is muuch faster...
    return cv2.warpPerspective(img, transform_matrix, (img_width, img_height))
    # return warp_perspective(img, transform_matrix, (img_width, img_height))

def bird_to_image(img, calibration):
    img_width, img_height = img.shape[0:2]
    src_rect = np.float32(calibration)
    dest_rect = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]]) 
    transform_matrix = get_perspective_transform(src_rect, dest_rect)
    return cv2.warpPerspective(img, transform_matrix, (img_width, img_height))
    # return warp_perspective(img, transform_matrix, (img_width, img_height))



def project_point_on_bird(img, calibration, p):
    img_width, img_height = img.shape[0:2]
    src_rect = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
    dest_rect = np.float32(calibration)
    transform_matrix = get_perspective_transform(src_rect, dest_rect)
    M = transform_matrix
    px = (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
    py = (M[1][0] * p[0] + M[1][1] * p[1] + M[1][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
    return (int(px), int(py))

def project_point_on_image(img, calibration, p):
    img_width, img_height = img.shape[0:2]
    src_rect = np.float32(calibration)
    dest_rect = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]]) 
    transform_matrix = get_perspective_transform(src_rect, dest_rect)
    M = transform_matrix
    px = (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
    py = (M[1][0] * p[0] + M[1][1] * p[1] + M[1][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
    return (int(px), int(py))


def distance(p1, p2):
    # calculates the euclidean distance between two centroids...
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def centroids(boxes, image, bird_view, calibration):
    # Returns the center of each bounding box in bird eye view...
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

def get_results(img, calibration, bounding_boxes):
    img_bird = image_to_bird(img, calibration)

    _centroids = centroids(bounding_boxes, img, img_bird, calibration)

    distances = []

    for i in range(0, len(_centroids)):
        ith_distances = []
        for j in range(0, len(_centroids)):
            ith_distances.append(distance(_centroids[i], _centroids[j]))
        distances.append(ith_distances)

    return distances    