import cv2, numpy as np
from params import bounding_boxes, frame
from transformations import get_perspective_transform, warp_perspective, project_point
from PIL import Image
from matplotlib import pyplot as plt



def image_to_bird(img, calibration):
    img_height,img_width = img.shape[0:2]
    src_rect = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
    dest_rect = np.float32(calibration)
    transform_matrix = get_perspective_transform(src_rect, dest_rect)
    # cv2.warpPerspective is muuch faster...
    return cv2.warpPerspective(img, transform_matrix, (img_width, img_height))
    # return warp_perspective(img, transform_matrix, (img_width, img_height))

def bird_to_image(img, calibration):
    img_height,img_width = img.shape[0:2]
    src_rect = np.float32(calibration)
    dest_rect = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]]) 
    transform_matrix = get_perspective_transform(src_rect, dest_rect)
    return cv2.warpPerspective(img, transform_matrix, (img_width, img_height))
    # return warp_perspective(img, transform_matrix, (img_width, img_height))



def project_point_on_bird(img, calibration, p):
    img_height,img_width = img.shape[0:2]
    src_rect = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
    dest_rect = np.float32(calibration)
    transform_matrix = cv2.getPerspectiveTransform(src_rect, dest_rect)
    return project_point(p[0], p[1], transform_matrix)

def project_point_on_image(img, calibration, p):
    img_height,img_width = img.shape[0:2]
    src_rect = np.float32(calibration)
    dest_rect = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]]) 
    transform_matrix = cv2.getPerspectiveTransform(src_rect, dest_rect)
    return project_point(p[0], p[1], transform_matrix)


def distance(p1, p2):
    # calculates the euclidean distance between two centroids...
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def centroids(boxes, image, bird_view, calibration):
    # Returns the center of each bounding box in bird eye view...
    centroids = []
    for b in boxes:
        xmin, ymax, xmax, ymin = b[0], b[1], b[2], b[3]
        w = xmax - xmin
        h = ymax - ymin
        x = xmin + w/2
        y = ymax - h/2
        # bird_x, bird_y = project_point_on_bird(bird_view, calibration, (x, ymax))
        center_bird_x, center_bird_y = project_point_on_bird(image, calibration, (x, ymin))
        centroids.append([int(center_bird_x), int(center_bird_y)])

    return centroids


def get_transformed_ellipse(img, calibration, center_bird, circle_radius, bbox):
    
    # Compute axis...
    max_r = center_bird[0] + circle_radius 
    max_c = center_bird[1] + circle_radius
    max_r_pt =  project_point_on_image(img, calibration, [max_r, center_bird[1]])
    max_c_pt =  project_point_on_image(img, calibration, [center_bird[0], max_c])
    center_img = project_point_on_image(img, calibration, center_bird)
    r_radius = distance(center_img, max_r_pt)
    c_radius = distance(center_img, max_c_pt)
    return {
        "r": center_img[0],
        "c": center_img[1],
        "r_radius": r_radius,
        "c_radius": c_radius,
    }

def get_distances_for_bounding_box(centroid, _centroids):
    distances = []

    for _centroid in _centroids:
        distances.append(distance(centroid, _centroid))

    return distances

def get_coordinates_with_distances(img, calibration, bounding_boxes, circle_radius=3):
    img_bird = image_to_bird(img, calibration)

    _centroids = centroids(bounding_boxes, img, img_bird, calibration)
    bboxes = []
    data = {
        "warped_size": [0, 0]
    }
    for i in range(0, len(bounding_boxes)):
        bbox_data = {}
        bbox_data["bird_centroid"] = _centroids[i]
        bbox_data["distances"] = get_distances_for_bounding_box(bbox_data["bird_centroid"], _centroids)
        bbox_data["circle_radius"] = circle_radius
        bbox_data["transformed_ellipse_params"] = get_transformed_ellipse(img, calibration, bbox_data["bird_centroid"], circle_radius, bounding_boxes[i])
        bboxes.append(bbox_data)
    data["bboxes"] = bboxes
    return data