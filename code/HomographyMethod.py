import cv2
import numpy as np


def compute_homography_transform_matrix(point_pairs):
    source_points = np.array([point_pair[0] for point_pair in point_pairs])
    target_points = np.array([point_pair[1] for point_pair in point_pairs])

    source_points = np.array(source_points).reshape(-1, 1, 2)
    target_points = np.array(target_points).reshape(-1, 1, 2)

    transform_matrix, mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0)

    return transform_matrix

