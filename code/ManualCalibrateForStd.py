import numpy as np
from matplotlib import pyplot as plt

import GradientDescent
import HomographyMethod
import ICP
import UtilFunctions


def compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list: list, calibration_point_list_modified: list):
    point_pair_list = []
    for index in range(len(avg_gaze_coordinate_after_translation_list)):
        point_pair = [np.array(avg_gaze_coordinate_after_translation_list[index]), np.array(calibration_point_list_modified[index])]
        point_pair_list.append(point_pair)

    distance_list = []
    for point_pair_index, point_pair in enumerate(point_pair_list):
        std_point = point_pair[0]
        correction_point = point_pair[1]
        distance = np.linalg.norm(std_point - correction_point)  # TODO 这里计算distance的方法可以再做定义。
        distance_list.append(distance)
    avg_distance = np.mean(distance_list)
    return distance_list, avg_distance


def compute_std_cali_with_icp(subject_index, calibration_data):
    '''
    用于计算标准校准点（用户注视的300个校准点）的校准参数。
    :param subject_index:
    :param calibration_data:
    :return:
    '''

    gaze_list = calibration_data[subject_index][0]
    avg_gaze_list = calibration_data[subject_index][1]
    calibration_point_list = calibration_data[subject_index][2]

    point_pairs = UtilFunctions.get_paired_points_of_std_cali_from_cali_dict(avg_gaze_list, calibration_point_list)
    rotation_matrix, translation_x, translation_y = ICP.point_based_matching(point_pairs)

    rotation_matrix = np.array(rotation_matrix)
    translation_vector = np.array([translation_x, translation_y])

    transform_matrix = np.array([[rotation_matrix[0][0], rotation_matrix[0][1], translation_x],
                                 [rotation_matrix[1][0], rotation_matrix[1][1], translation_y],
                                 [0, 0, 1]])

    return transform_matrix


def compute_std_cali_with_rotation_and_translation_gradient_descent(subject_index, calibration_data):
    gaze_list = calibration_data[subject_index][0]
    avg_gaze_list = calibration_data[subject_index][1]
    calibration_point_list = calibration_data[subject_index][2]

    point_pairs = UtilFunctions.get_paired_points_of_std_cali_from_cali_dict(avg_gaze_list, calibration_point_list)
    transform_matrix = GradientDescent.gradient_descent_with_rotation_and_translation(point_pairs)

    return transform_matrix


def compute_std_cali_with_whole_matrix_gradient_descent(subject_index, calibration_data):
    gaze_list = calibration_data[subject_index][0]
    avg_gaze_list = calibration_data[subject_index][1]
    calibration_point_list = calibration_data[subject_index][2]

    point_pairs = UtilFunctions.get_paired_points_of_std_cali_from_cali_dict(avg_gaze_list, calibration_point_list)
    transform_matrix = GradientDescent.gradient_descent_with_whole_matrix_using_tensor(point_pairs)

    return transform_matrix


def compute_std_cali_with_homography_matrix(subject_index, calibration_data):
    gaze_list = calibration_data[subject_index][0]
    avg_gaze_list = calibration_data[subject_index][1]
    calibration_point_list = calibration_data[subject_index][2]

    point_pairs = UtilFunctions.get_paired_points_of_std_cali_from_cali_dict(avg_gaze_list, calibration_point_list)
    transform_matrix = HomographyMethod.compute_homography_transform_matrix(point_pairs)

    return transform_matrix


def apply_transform_to_calibration(subject_index, calibration_data, transform_matrix):
    gaze_list = calibration_data[subject_index][0]
    avg_gaze_list = calibration_data[subject_index][1]
    calibration_point_list = calibration_data[subject_index][2]

    gaze_coordinates_before_translation_list = []
    avg_gaze_coordinate_before_translation_list = []
    gaze_coordinates_after_translation_list = []
    avg_gaze_coordinate_after_translation_list = []
    calibration_point_list_modified = []
    for row_index in range(len(gaze_list)):
        for col_index in range(len(gaze_list[row_index])):
            gaze_dict = gaze_list[row_index][col_index]
            gaze_x_list = gaze_dict["gaze_x"]
            gaze_y_list = gaze_dict["gaze_y"]
            gaze_coordinates = [np.array([gaze_x_list[i], gaze_y_list[i]]) for i in range(len(gaze_x_list))]
            gaze_coordinates_before_translation_list.append(np.array(gaze_coordinates))

            gaze_coordinates_after_translation = [np.dot(transform_matrix, UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_coordinate)) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates_after_translation = [UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates_after_translation]
            gaze_coordinates_after_translation_list.append(np.array(gaze_coordinates_after_translation))

            avg_gaze_dict = avg_gaze_list[row_index][col_index]
            avg_gaze_x = avg_gaze_dict["avg_gaze_x"]
            avg_gaze_y = avg_gaze_dict["avg_gaze_y"]
            avg_gaze_coordinate = np.array([avg_gaze_x, avg_gaze_y])
            avg_gaze_coordinate_before_translation_list.append(avg_gaze_coordinate)
            avg_gaze_coordinate_after_translation = np.dot(transform_matrix, UtilFunctions.change_2d_vector_to_homogeneous_vector(avg_gaze_coordinate))
            avg_gaze_coordinate_after_translation = UtilFunctions.change_homogeneous_vector_to_2d_vector(avg_gaze_coordinate_after_translation)
            avg_gaze_coordinate_after_translation_list.append(avg_gaze_coordinate_after_translation)

            calibration_point_dict = calibration_point_list[row_index][col_index]
            calibration_point_x = calibration_point_dict["point_x"]
            calibration_point_y = calibration_point_dict["point_y"]
            calibration_point = np.array([calibration_point_x, calibration_point_y])
            calibration_point_list_modified.append(calibration_point)

    gaze_coordinates_before_translation_list = np.array(gaze_coordinates_before_translation_list)
    gaze_coordinates_before_translation_list = np.concatenate(gaze_coordinates_before_translation_list, axis=0)

    gaze_coordinates_after_translation_list = np.array(gaze_coordinates_after_translation_list)
    gaze_coordinates_after_translation_list = np.concatenate(gaze_coordinates_after_translation_list, axis=0)

    avg_gaze_coordinate_before_translation_list = np.array(avg_gaze_coordinate_before_translation_list)
    avg_gaze_coordinate_after_translation_list = np.array(avg_gaze_coordinate_after_translation_list)

    calibration_point_list_modified = np.array(calibration_point_list_modified)

    return gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
           avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
           calibration_point_list_modified



