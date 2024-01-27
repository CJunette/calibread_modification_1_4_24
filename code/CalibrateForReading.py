import multiprocessing
import os.path
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import GradientDescent
import ManualCalibrateForStd
import Render
import UtilFunctions
import configs
from scipy.spatial import ConvexHull
from math import atan2, degrees

def convert_data_format(reading_data, text_data):
    text_point_list = []
    gaze_point_list = []
    original_gaze_list = []
    gaze_density_list = []
    for text_index in range(len(reading_data)):
        text_x_list = text_data[text_index]["x"].values.tolist()
        text_y_list = text_data[text_index]["y"].values.tolist()
        text_points = [np.array([text_x_list[i], text_y_list[i]]) for i in range(len(text_x_list))]
        text_point_list.append(text_points)
        reading_df = reading_data[text_index]
        gaze_x_list = reading_df["gaze_x"].values.tolist()
        gaze_y_list = reading_df["gaze_y"].values.tolist()
        gaze_points = [np.array([gaze_x_list[i], gaze_y_list[i]]) for i in range(len(gaze_x_list))]
        gaze_point_list.append(gaze_points)
        density_list = reading_df["density"].values.tolist()
        gaze_density_list.append(density_list)
        original_gaze_points = [np.array([gaze_x_list[i], gaze_y_list[i]]) for i in range(len(gaze_x_list))]
        original_gaze_list.append(original_gaze_points)

    return text_point_list, gaze_point_list, original_gaze_list, gaze_density_list


def find_nearest_neighbor(point_list):
    nbrs_list = []
    for text_index in range(len(point_list)):
        text_points = point_list[text_index]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(text_points)  # TODO 目前先使用已有的、基于物理距离的算法来进行point_pair的匹配，之后需要添加语义时，可能要改写一下这里的方法。
        nbrs_list.append(nbrs)

    return nbrs_list


def calibrate_with_location(subject_index, reading_data, text_data, calibration_data, max_iteration=100, distance_threshold=500):
    text_point_list, gaze_point_list, original_gaze_list, gaze_density_list = convert_data_format(reading_data, text_data)
    nbrs_list = find_nearest_neighbor(text_point_list)

    total_transform_matrix = np.eye(3)
    avg_error_list = []
    for iteration_index in range(max_iteration):
        print("iteration_index: ", iteration_index)
        point_pair_list = []
        for text_index in range(len(text_point_list)):
            distances, indices = nbrs_list[text_index].kneighbors(gaze_point_list[text_index])
            for nn_index in range(len(distances)):
                if distances[nn_index][0] < distance_threshold:
                    point_pair_list.append((gaze_point_list[text_index][nn_index], text_point_list[text_index][indices[nn_index][0]]))

        transform_matrix = GradientDescent.gradient_descent_with_whole_matrix_using_tensor(point_pair_list)
        # update total_transform_matrix
        total_transform_matrix = np.dot(transform_matrix, total_transform_matrix)

        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
            avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
            calibration_point_list_modified = ManualCalibrateForStd.apply_transform_to_calibration(subject_index, calibration_data, total_transform_matrix)

        # TODO 这里先简单写一个看效果的demo，之后再将函数做合适的封装处理。
        distance_list, avg_distance = ManualCalibrateForStd.compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        avg_error_list.append(avg_distance)
        print("average distance: ", avg_distance)

        # update points.
        for text_index in range(len(text_point_list)):
            for gaze_index in range(len(gaze_point_list[text_index])):
                old_gaze_coordinates = gaze_point_list[text_index][gaze_index]
                new_gaze_coordinates = np.dot(transform_matrix, UtilFunctions.change_2d_vector_to_homogeneous_vector(old_gaze_coordinates))
                new_gaze_coordinates = UtilFunctions.change_homogeneous_vector_to_2d_vector(new_gaze_coordinates)
                gaze_point_list[text_index][gaze_index] = new_gaze_coordinates

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)
        ax.set_aspect("equal")
        for text_index in range(len(text_point_list)):
            ax.scatter(np.array(gaze_point_list[text_index])[:, 0], np.array(gaze_point_list[text_index])[:, 1], c='g', marker='o', s=1)
            ax.scatter(np.array(original_gaze_list[text_index])[:, 0], np.array(original_gaze_list[text_index])[:, 1], c='b', marker='o', s=1)
            ax.scatter(np.array(text_point_list[text_index])[:, 0], np.array(text_point_list[text_index])[:, 1], c='k', marker='o')

        plt.show()

        # Render.visualize_cali_result(gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list,
        #                              avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list,
        #                              calibration_point_list_modified)

    for iteration_index in range(len(avg_error_list)):
        print("avg_error_list[", iteration_index, "]: ", avg_error_list[iteration_index])

    return avg_error_list


def calibrate_with_location_and_coverage(subject_index, reading_data, text_data, calibration_data, max_iteration=100, distance_threshold=500):
    '''
    在这里我们假设当阅读数量足够多时，每个文本点都至少会有一个对应的阅读点。
    实现这一步的方法是利用一个字典收集所有有效的文本点（即该文本点至少显示过文字），然后在每次迭代中，用另一个字典查看是否每个文本点都被阅读点覆盖。
    :param subject_index:
    :param reading_data:
    :param text_data:
    :param calibration_data:
    :param max_iteration:
    :param distance_threshold:
    :return:
    '''
    text_point_list, gaze_point_list, original_gaze_list, gaze_density_list = convert_data_format(reading_data, text_data)
    nbrs_list = find_nearest_neighbor(text_point_list)

    # 用一个dict来记录所有有效的文本点。
    text_point_dict = {}
    for row_index in range(len(calibration_data[subject_index][2])):
        for col_index in range(len(calibration_data[subject_index][2][row_index])):
            x = calibration_data[subject_index][2][row_index][col_index]["point_x"]
            y = calibration_data[subject_index][2][row_index][col_index]["point_y"]
            text_point_dict[(x, y)] = 0
    for text_index in range(len(text_point_list)):
        for point_index in range(len(text_point_list[text_index])):
            x = int(text_point_list[text_index][point_index][0])
            y = int(text_point_list[text_index][point_index][1])
            if (x, y) in text_point_dict:
                text_point_dict[(x, y)] += 1
    effective_text_point_dict = {}
    for key in text_point_dict:
        if text_point_dict[key] != 0:
            effective_text_point_dict[key] = text_point_dict[key]
    text_point_total_utilized_count = 0
    for key in effective_text_point_dict:
        text_point_total_utilized_count += effective_text_point_dict[key]

    total_transform_matrix = np.eye(3)
    avg_error_list = []
    for iteration_index in range(max_iteration):
        print("iteration_index: ", iteration_index)
        # 没次迭代前，创建一个类似effective_text_point_dict的字典，用于记录每个文本点被阅读点覆盖的次数。
        actual_text_point_dict = effective_text_point_dict.copy()
        for key in actual_text_point_dict:
            actual_text_point_dict[key] = 0

        point_pair_list = []
        for text_index in range(len(text_point_list)):
            distances, indices = nbrs_list[text_index].kneighbors(gaze_point_list[text_index])
            for nn_index in range(len(distances)):
                if distances[nn_index][0] < distance_threshold:
                    point_pair_list.append([gaze_point_list[text_index][nn_index], text_point_list[text_index][indices[nn_index][0]]])
                    x = int(text_point_list[text_index][indices[nn_index][0]][0])
                    y = int(text_point_list[text_index][indices[nn_index][0]][1])
                    if (x, y) in actual_text_point_dict:
                        actual_text_point_dict[(x, y)] += 1

        # 如果某个文本点没有被阅读点覆盖，则需要从所有的阅读点中，找到前n个最近的阅读点，然后将这些阅读点与该文本点进行匹配，同时删除这些阅读点与之前匹配点的匹配关系。n由这个阅读点的实际使用次数与所有阅读点的实际使用次数之和的比值乘以阅读点总数决定。
        for key in actual_text_point_dict:
            if actual_text_point_dict[key] == 0:
                closet_point_num = int(len(point_pair_list) * effective_text_point_dict[key] / text_point_total_utilized_count)

                text_point_x = float(key[0])
                text_point_y = float(key[1])

                distance_list = []
                for point_pair_index in range(len(point_pair_list)):
                    gaze_point = point_pair_list[point_pair_index][0]
                    distance = np.linalg.norm(np.array([text_point_x, text_point_y]) - gaze_point)
                    distance_list.append((distance, point_pair_index))

                distance_list.sort(key=lambda x: x[0])

                for point_index in range(closet_point_num):
                    closet_point_index = distance_list[point_index][1]
                    point_pair_list[closet_point_index][1] = np.array([text_point_x, text_point_y])

        transform_matrix = GradientDescent.gradient_descent_with_whole_matrix_using_tensor(point_pair_list, max_iterations=1000)
        # update total_transform_matrix
        total_transform_matrix = np.dot(transform_matrix, total_transform_matrix)

        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
            avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
            calibration_point_list_modified = ManualCalibrateForStd.apply_transform_to_calibration(subject_index, calibration_data, total_transform_matrix)

        # TODO 这里先简单写一个看效果的demo，之后再将函数做合适的封装处理。
        distance_list, avg_distance = ManualCalibrateForStd.compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        avg_error_list.append(avg_distance)
        print("average distance: ", avg_distance)

        # update points.
        for text_index in range(len(text_point_list)):
            for gaze_index in range(len(gaze_point_list[text_index])):
                old_gaze_coordinates = gaze_point_list[text_index][gaze_index]
                new_gaze_coordinates = np.dot(transform_matrix, UtilFunctions.change_2d_vector_to_homogeneous_vector(old_gaze_coordinates))
                new_gaze_coordinates = UtilFunctions.change_homogeneous_vector_to_2d_vector(new_gaze_coordinates)
                gaze_point_list[text_index][gaze_index] = new_gaze_coordinates

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)
        ax.set_aspect("equal")
        for text_index in range(len(text_point_list)):
            ax.scatter(np.array(gaze_point_list[text_index])[:, 0], np.array(gaze_point_list[text_index])[:, 1], c='g', marker='o', s=1)
            ax.scatter(np.array(original_gaze_list[text_index])[:, 0], np.array(original_gaze_list[text_index])[:, 1], c='b', marker='o', s=1)
            ax.scatter(np.array(text_point_list[text_index])[:, 0], np.array(text_point_list[text_index])[:, 1], c='k', marker='o')
        plt.show()

        # Render.visualize_cali_result(gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list,
        #                              avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list,
        #                              calibration_point_list_modified)

    for iteration_index in range(len(avg_error_list)):
        print("avg_error_list[", iteration_index, "]: ", avg_error_list[iteration_index])

    return avg_error_list


def calibrate_with_location_coverage_and_penalty(subject_index, reading_data, text_data, calibration_data, max_iteration=100, distance_threshold=64):
    '''
    在这里我们假设当阅读数量足够多时，每个文本点都至少会有一个对应的阅读点。
    实现这一步的方法是利用一个字典收集所有有效的文本点（即该文本点至少显示过文字），然后在每次迭代中，用另一个字典查看是否每个文本点都被阅读点覆盖。
    此外，我们假设每个文本点都有一个惩罚值，这个惩罚值由该文本点具体位置，及其是否为标点或空格决定。这个惩罚值最终被放到gradient descent中，作为一个error距离的乘积项。
    :param subject_index:
    :param reading_data:
    :param text_data:
    :param calibration_data:
    :param max_iteration:
    :param distance_threshold:
    :return:
    '''
    text_point_list, gaze_point_list, original_gaze_list, gaze_density_list = convert_data_format(reading_data, text_data)
    nbrs_text_list = find_nearest_neighbor(text_point_list)

    total_gaze_point_num = 0

    # 创建一个包含gaze_point坐标、所属text_index及在gaze_point_list[text_index]中序号的list，用于后续的惩罚项修改时的计算。
    gaze_point_info_list = []
    for text_index in range(len(gaze_point_list)):
        for gaze_index in range(len(gaze_point_list[text_index])):
            density = gaze_density_list[text_index][gaze_index]
            row_label = reading_data[text_index].iloc[gaze_index]["row_label"]
            gaze_point_info_list.append([gaze_point_list[text_index][gaze_index], text_index, gaze_index, density, row_label])
            total_gaze_point_num += 1

    gaze_point_list_1d = []
    for text_index in range(len(gaze_point_list)):
        for gaze_index in range(len(gaze_point_list[text_index])):
            gaze_point_list_1d.append(gaze_point_list[text_index][gaze_index])

    # 用一个dict来记录所有有效的文本点。
    text_point_dict = {}
    for row_index in range(len(calibration_data[subject_index][2])):
        for col_index in range(len(calibration_data[subject_index][2][row_index])):
            x = calibration_data[subject_index][2][row_index][col_index]["point_x"]
            y = calibration_data[subject_index][2][row_index][col_index]["point_y"]
            text_point_dict[(x, y)] = 0
    for text_index in range(len(text_data)):
        for index, row in text_data[text_index].iterrows():
            x = row["x"]
            y = row["y"]
            word = row["word"]
            if word != "blank_supplement" and (x, y) in text_point_dict:
                text_point_dict[(x, y)] += 1

    effective_text_point_dict = {}
    for key in text_point_dict:
        if text_point_dict[key] != 0:
            effective_text_point_dict[key] = text_point_dict[key]
    text_point_total_utilized_count = 0
    for key in effective_text_point_dict:
        text_point_total_utilized_count += effective_text_point_dict[key]

    supplement_text_point_dict = {}
    for text_index in range(len(text_data)):
        for index, row in text_data[text_index].iterrows():
            x = row["x"]
            y = row["y"]
            word = row["word"]
            if word == "blank_supplement":
                supplement_text_point_dict[(x, y)] = 0

    # 读取每个文本点的惩罚值。
    text_penalty_list_1 = []
    for text_index in range(len(text_data)):
        df = text_data[text_index]
        text_penalty_list_2 = []
        for index, row in df.iterrows():
            penalty = row["penalty"]
            text_penalty_list_2.append(penalty)
        text_penalty_list_1.append(text_penalty_list_2)

    # 将gaze data和text point先做一个基于重心的对齐。
    gaze_point_center = np.mean(gaze_point_list_1d, axis=0)
    text_point_center = np.array([0, 0])
    for key, value in effective_text_point_dict.items():
        text_point_center[0] += key[0]
        text_point_center[1] += key[1]
    text_point_center[0] /= len(effective_text_point_dict)
    text_point_center[1] /= len(effective_text_point_dict)
    translate_vector = np.array(text_point_center - gaze_point_center)
    for gaze_index in range(len(gaze_point_list_1d)):
        gaze_point_list_1d[gaze_index] += translate_vector
    # 把calibration的数据也做一下相同的移动。
    for row_index in range(len(calibration_data[subject_index][1])):
        for col_index in range(len(calibration_data[subject_index][1][row_index])):
            calibration_data[subject_index][1][row_index][col_index]["avg_gaze_x"] += translate_vector[0]
            calibration_data[subject_index][1][row_index][col_index]["avg_gaze_y"] += translate_vector[1]
            for point_index in range(len(calibration_data[subject_index][0][row_index][col_index]["gaze_x"])):
                calibration_data[subject_index][0][row_index][col_index]["gaze_x"][point_index] += translate_vector[0]
                calibration_data[subject_index][0][row_index][col_index]["gaze_y"][point_index] += translate_vector[1]

    total_transform_matrix = np.eye(3)
    avg_error_list = []
    last_gd_error = 100000
    for iteration_index in range(max_iteration):
        print("iteration_index: ", iteration_index)
        # 每次迭代前，创建一个类似effective_text_point_dict的字典，用于记录每个文本点被阅读点覆盖的次数。
        actual_text_point_dict = effective_text_point_dict.copy()
        for key in actual_text_point_dict:
            actual_text_point_dict[key] = 0

        actual_supplement_text_point_dict = supplement_text_point_dict.copy()

        gaze_point_nbrs = NearestNeighbors(n_neighbors=int(total_gaze_point_num/5), algorithm='kd_tree').fit(gaze_point_list_1d)

        point_pair_list = []
        weight_list = []
        point_pair_num = 0

        for text_index in range(len(text_point_list)):
            distances, indices = nbrs_text_list[text_index].kneighbors(gaze_point_list[text_index])
            for gaze_index in range(len(distances)):
                if distances[gaze_index][0] < distance_threshold:
                    point_pair_list.append([gaze_point_list[text_index][gaze_index], text_point_list[text_index][indices[gaze_index][0]]])
                    density = gaze_density_list[text_index][gaze_index]
                    weight = text_penalty_list_1[text_index][indices[gaze_index][0]] * density * 0.1
                    weight_list.append(weight)
                    point_pair_num += 1

                    x = text_point_list[text_index][indices[gaze_index][0]][0]
                    y = text_point_list[text_index][indices[gaze_index][0]][1]
                    if (x, y) in actual_text_point_dict:
                        actual_text_point_dict[(x, y)] += 1
                    if (x, y) in actual_supplement_text_point_dict:
                        actual_supplement_text_point_dict[(x, y)] += 1

        # 如果某个文本点没有被阅读点覆盖，则需要从所有的阅读点中，找到前n个最近的阅读点，然后将这些阅读点与该文本点进行匹配，同时删除这些阅读点与之前匹配点的匹配关系。n由这个阅读点的实际使用次数与所有阅读点的实际使用次数之和的比值乘以阅读点总数决定。
        # 我觉得这里的匹配关系在后续的迭代中会出一些问题，当匹配逐渐稳定时，这些“补充配对点”可能会导致匹配关系的不稳定。所以更好的做法可能是在一开始就先按重心做一个对齐。（试了重心对齐，不大行，会导致gaze区域不断缩小。）
        for key in actual_text_point_dict:
            if actual_text_point_dict[key] == 0:
                closet_point_num = int(point_pair_num * effective_text_point_dict[key] / text_point_total_utilized_count)
                cur_text_point = [float(key[0]), float(key[1])]
                distances, indices = gaze_point_nbrs.kneighbors([cur_text_point])
                for point_index in range(closet_point_num):
                    current_point_index = indices[0][point_index]
                    closet_point_info = gaze_point_list_1d[current_point_index]
                    if current_point_index < len(point_pair_list):
                        point_pair_list[current_point_index] = [closet_point_info, cur_text_point]
                        weight_list[current_point_index] = 1 # 目前暂时先让这些“补充配对点”不参与惩罚项的计算。
                    else:
                        point_pair_list.append([closet_point_info, cur_text_point])
                        weight_list.append(1)

        transform_matrix, gd_error = GradientDescent.gradient_descent_with_whole_matrix_using_tensor_with_weight(point_pair_list, weight_list, max_iterations=1000)

        # TODO 这里出了点小情况：虽然gradient descent的下降带来了正确的效果，但transform后的匹配可能产生错误，使得后一轮的error甚至要高于前一轮的error。这里先简单地用一个if语句来解决这个问题，之后再想想更好的办法。
        # if gd_error < last_gd_error:
        last_gd_error = gd_error
        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
            avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
            calibration_point_list_modified = ManualCalibrateForStd.apply_transform_to_calibration(subject_index, calibration_data, total_transform_matrix)

        # update gaze_point_list_1d
        gaze_point_list_1d = [UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_point) for gaze_point in gaze_point_list_1d]
        gaze_point_list_1d = [np.dot(transform_matrix, gaze_point) for gaze_point in gaze_point_list_1d]
        gaze_point_list_1d = [UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_point_list_1d]
        # update total_transform_matrix
        total_transform_matrix = np.dot(transform_matrix, total_transform_matrix)

        # TODO 这里先简单写一个看效果的demo，之后再将函数做合适的封装处理。
        distance_list, avg_distance = ManualCalibrateForStd.compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        avg_error_list.append(avg_distance)
        print("average distance: ", avg_distance)

        fig = plt.figure(figsize=(24, 12))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1920)
        ax.set_ylim(800, 0)
        ax.set_aspect("equal")
        for text_index in range(len(text_point_list)):
            ax.scatter(np.array(gaze_point_list[text_index])[:, 0], np.array(gaze_point_list[text_index])[:, 1], c='orange', marker='o', s=1, zorder=1)
            # ax.scatter(np.array(original_gaze_list[text_index])[:, 0], np.array(original_gaze_list[text_index])[:, 1], c='b', marker='o', s=1)

        # update points.
        for text_index in range(len(text_point_list)):
            for gaze_index in range(len(gaze_point_list[text_index])):
                old_gaze_coordinates = gaze_point_list[text_index][gaze_index]
                new_gaze_coordinates = np.dot(transform_matrix, UtilFunctions.change_2d_vector_to_homogeneous_vector(old_gaze_coordinates))
                new_gaze_coordinates = UtilFunctions.change_homogeneous_vector_to_2d_vector(new_gaze_coordinates)
                gaze_point_list[text_index][gaze_index] = new_gaze_coordinates

        for text_index in range(len(text_point_list)):
            ax.scatter(np.array(gaze_point_list[text_index])[:, 0], np.array(gaze_point_list[text_index])[:, 1], c='g', marker='o', s=1, zorder=1)
            # ax.scatter(np.array(original_gaze_list[text_index])[:, 0], np.array(original_gaze_list[text_index])[:, 1], c='b', marker='o', s=1)

        max_pair_num = max(actual_text_point_dict.values())
        for key, value in actual_text_point_dict.items():
            color_ratio = 0.8 - (value / max_pair_num) * 0.6
            color = (color_ratio, color_ratio, color_ratio)
            if value == 0:
                ax.scatter(key[0], key[1], c=color, marker='x', s=40, zorder=2)
            else:
                ax.scatter(key[0], key[1], c=color, marker='o', s=40, zorder=2)

        max_pair_num = max(actual_supplement_text_point_dict.values())
        for key, value in actual_supplement_text_point_dict.items():
            color_ratio = 0.8 - (value / max_pair_num) * 0.6
            color = (1, color_ratio, color_ratio)
            if value == 0:
                ax.scatter(key[0], key[1], c=color, marker='x', s=10, zorder=3)
            else:
                ax.scatter(key[0], key[1], c=color, marker='o', s=10, zorder=3)

        line_segment_list = []
        for point_pair_index in range(len(point_pair_list)):
            # point_pair_gaze = UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair_list[point_pair_index][0])
            # point_pair_gaze = np.dot(transform_matrix, point_pair_gaze)
            # point_pair_gaze = UtilFunctions.change_homogeneous_vector_to_2d_vector(point_pair_gaze)
            point_pair_gaze = point_pair_list[point_pair_index][0]
            line_segment_list.append([point_pair_gaze, point_pair_list[point_pair_index][1]])
        line_collection = LineCollection(line_segment_list, colors='b', linewidths=0.5, zorder=0)
        ax.add_collection(line_collection)

        plt.show()

        Render.visualize_cali_result(gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list,
                                     avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list,
                                     calibration_point_list_modified)

    for iteration_index in range(len(avg_error_list)):
        print("avg_error_list[", iteration_index, "]: ", avg_error_list[iteration_index])

    return avg_error_list


def single_process_matching(reading_data, text_data,
                            text_index, row_index,
                            filtered_text_data_list,
                            total_nbrs_list, row_nbrs_list,
                            actual_text_point_dict, actual_supplement_text_point_dict,
                            distance_threshold):
    reading_df = reading_data[text_index]
    filtered_reading_df = reading_df[reading_df["row_label"] == row_index]

    point_pair_list = []
    weight_list = []
    row_label_list = []

    # 如果当前行没有text point或当前reading数据没有一个gaze point被分到了这一行，则对于这里的reading point，需要匹配所有的text point。
    if row_nbrs_list[text_index][row_index] is None or filtered_reading_df.shape[0] == 0:
        reading_coordinates = reading_df[["gaze_x", "gaze_y"]].values.tolist()
        distances, indices = total_nbrs_list[text_index].kneighbors(reading_coordinates)
        text_coordinate = text_data[text_index][["x", "y"]].values.tolist()
        for gaze_index in range(len(distances)):
            if distances[gaze_index][0] < distance_threshold:
                point_pair_list.append([reading_coordinates[gaze_index], text_coordinate[indices[gaze_index][0]]])
                weight = text_data[text_index].iloc[indices[gaze_index][0]]["penalty"]
                weight_list.append(weight)
                row_label_list.append(-1)
                x = text_data[text_index].iloc[indices[gaze_index][0]]["x"]
                y = text_data[text_index].iloc[indices[gaze_index][0]]["y"]
                if (x, y) in actual_text_point_dict:
                    actual_text_point_dict[(x, y)] += 1
                if (x, y) in actual_supplement_text_point_dict:
                    actual_supplement_text_point_dict[(x, y)] += 1
    # 如果当前行有text point且当前reading数据至少有一个gaze point被分到了这一行，则对于这里的reading point，只需要匹配当前行的text point。
    else:
        filtered_reading_coordinates = filtered_reading_df[["gaze_x", "gaze_y"]].values.tolist()
        distances, indices = row_nbrs_list[text_index][row_index].kneighbors(filtered_reading_coordinates)
        filtered_text_data = filtered_text_data_list[text_index][row_index]
        filtered_text_coordinate = filtered_text_data[["x", "y"]].values.tolist()
        for gaze_index in range(len(distances)):
            if distances[gaze_index][0] < distance_threshold:
                point_pair_list.append([filtered_reading_coordinates[gaze_index], filtered_text_coordinate[indices[gaze_index][0]]])
                weight = filtered_text_data.iloc[indices[gaze_index][0]]["penalty"]
                weight_list.append(weight)
                row_label_list.append(row_index)
                x = filtered_text_data.iloc[indices[gaze_index][0]]["x"]
                y = filtered_text_data.iloc[indices[gaze_index][0]]["y"]
                if (x, y) in actual_text_point_dict:
                    actual_text_point_dict[(x, y)] += 1
                if (x, y) in actual_supplement_text_point_dict:
                    actual_supplement_text_point_dict[(x, y)] += 1

    return point_pair_list, weight_list, row_label_list


def point_matching_1(reading_data, text_data, filtered_text_data_list,
                     total_nbrs_list, row_nbrs_list,
                     actual_text_point_dict, actual_supplement_text_point_dict,
                     distance_threshold):
    point_pair_list = []
    weight_list = []
    row_label_list = []
    for text_index in range(len(reading_data)):
        reading_df = reading_data[text_index]
        for row_index in range(configs.row_num):
            filtered_reading_df = reading_df[reading_df["row_label"] == row_index]

            # 如果当前行没有text point或当前reading数据没有一个gaze point被分到了这一行，则对于这里的reading point，需要匹配所有的text point。
            if row_nbrs_list[text_index][row_index] is None or filtered_reading_df.shape[0] == 0:
                reading_coordinates = reading_df[["gaze_x", "gaze_y"]].values.tolist()
                distances, indices = total_nbrs_list[text_index].kneighbors(reading_coordinates)
                text_coordinate = text_data[text_index][["x", "y"]].values.tolist()
                for gaze_index in range(len(distances)):
                    if distances[gaze_index][0] < distance_threshold:
                        point_pair_list.append([reading_coordinates[gaze_index], text_coordinate[indices[gaze_index][0]]])
                        weight = text_data[text_index].iloc[indices[gaze_index][0]]["penalty"]
                        weight_list.append(weight)
                        row_label_list.append(-1)
                        x = text_data[text_index].iloc[indices[gaze_index][0]]["x"]
                        y = text_data[text_index].iloc[indices[gaze_index][0]]["y"]
                        if (x, y) in actual_text_point_dict:
                            actual_text_point_dict[(x, y)] += 1
                        if (x, y) in actual_supplement_text_point_dict:
                            actual_supplement_text_point_dict[(x, y)] += 1
            # 如果当前行有text point且当前reading数据至少有一个gaze point被分到了这一行，则对于这里的reading point，只需要匹配当前行的text point。
            else:
                filtered_reading_coordinates = filtered_reading_df[["gaze_x", "gaze_y"]].values.tolist()
                distances, indices = row_nbrs_list[text_index][row_index].kneighbors(filtered_reading_coordinates)
                filtered_text_data = filtered_text_data_list[text_index][row_index]
                filtered_text_coordinate = filtered_text_data[["x", "y"]].values.tolist()
                for gaze_index in range(len(distances)):
                    if distances[gaze_index][0] < distance_threshold:
                        point_pair_list.append([filtered_reading_coordinates[gaze_index], filtered_text_coordinate[indices[gaze_index][0]]])
                        weight = filtered_text_data.iloc[indices[gaze_index][0]]["penalty"]
                        weight_list.append(weight)
                        row_label_list.append(row_index)
                        x = filtered_text_data.iloc[indices[gaze_index][0]]["x"]
                        y = filtered_text_data.iloc[indices[gaze_index][0]]["y"]
                        if (x, y) in actual_text_point_dict:
                            actual_text_point_dict[(x, y)] += 1
                        if (x, y) in actual_supplement_text_point_dict:
                            actual_supplement_text_point_dict[(x, y)] += 1

    return point_pair_list, weight_list, row_label_list


def point_matching_2(reading_data, gaze_point_list_1d, text_data, filtered_text_data_list,
                     total_nbrs_list, row_nbrs_list,
                     effective_text_point_dict, actual_text_point_dict, actual_supplement_text_point_dict,
                     distance_threshold, ratio=0.1):
    # 首先遍历所有的reading point，找到与其label一致的text point，然后将这些匹配点加入到point_pair_list中。
    point_pair_list = []
    weight_list = []
    row_label_list = []
    for text_index in range(len(reading_data)):
        reading_df = reading_data[text_index]
        for row_index in range(configs.row_num):
            filtered_reading_df = reading_df[reading_df["row_label"] == row_index]

            # point_pair_list_1 = []
            # weight_list_1 = []
            # row_label_list_1 = []
            # distance_list_1 = []
            # total_reading_coordinates = reading_df[["gaze_x", "gaze_y"]].values.tolist()
            # distances, indices = total_nbrs_list[text_index].kneighbors(total_reading_coordinates)
            # text_coordinate = text_data[text_index][["x", "y"]].values.tolist()
            # for gaze_index in range(len(distances)):
            #     # if distances[gaze_index][0] < distance_threshold:
            #         point_pair_list_1.append([total_reading_coordinates[gaze_index], text_coordinate[indices[gaze_index][0]]])
            #         weight = text_data[text_index].iloc[indices[gaze_index][0]]["penalty"]
            #         weight_list_1.append(weight)
            #         row_label_list_1.append(-1)
            #         distance_list_1.append(distances[gaze_index][0])

            point_pair_list_2 = []
            weight_list_2 = []
            row_label_list_2 = []
            distance_list_2 = []
            if row_nbrs_list[text_index][row_index] and filtered_reading_df.shape[0] != 0:
                filtered_reading_coordinates = filtered_reading_df[["gaze_x", "gaze_y"]].values.tolist()
                filtered_distances_of_row, filtered_indices_of_row = row_nbrs_list[text_index][row_index].kneighbors(filtered_reading_coordinates)
                filtered_text_data = filtered_text_data_list[text_index][row_index]
                filtered_text_coordinate = filtered_text_data[["x", "y"]].values.tolist()

                # total_reading_coordinates = reading_df[["gaze_x", "gaze_y"]].values.tolist()
                # distances, indices = total_nbrs_list[text_index].kneighbors(total_reading_coordinates)
                text_coordinate = text_data[text_index][["x", "y"]].values.tolist()
                filtered_distance_of_all_text, filtered_indices_of_all_text = total_nbrs_list[text_index].kneighbors(filtered_reading_coordinates)

                for gaze_index in range(len(filtered_distances_of_row)):
                    if filtered_distances_of_row[gaze_index][0] < distance_threshold:
                        point_pair_list_2.append([filtered_reading_coordinates[gaze_index], filtered_text_coordinate[filtered_indices_of_row[gaze_index][0]]])
                        weight = filtered_text_data.iloc[filtered_indices_of_row[gaze_index][0]]["penalty"]
                        weight_list_2.append(weight)
                        row_label_list_2.append(row_index)
                        distance_list_2.append(filtered_distances_of_row[gaze_index][0])

                        # if filtered_distances_of_row[gaze_index][0] > distance_threshold / 2:
                        #     point_pair_list_2.append([filtered_reading_coordinates[gaze_index], text_coordinate[overall_indices[gaze_index][0]]])
                        #     weight = text_data[text_index].iloc[overall_indices[gaze_index][0]]["penalty"]
                        #     weight_list_2.append(weight)
                        #     row_label_list_2.append(-1)
                        #     distance_list_2.append(overall_distance[gaze_index][0])
                    else:
                        point_pair_list_2.append([filtered_reading_coordinates[gaze_index], text_coordinate[filtered_indices_of_all_text[gaze_index][0]]])
                        weight = text_data[text_index].iloc[filtered_indices_of_all_text[gaze_index][0]]["penalty"]
                        weight_list_2.append(weight)
                        row_label_list_2.append(row_index)
                        distance_list_2.append(filtered_distance_of_all_text[gaze_index][0])

            # for gaze_index in range(len(point_pair_list_1)):
            #     # if distance_list_1[gaze_index] < distance_list_2[gaze_index] * ratio:
            #     #     point_pair_list_2[gaze_index] = point_pair_list_1[gaze_index]
            #     #     weight_list_2[gaze_index] = weight_list_1[gaze_index]
            #     #     row_label_list_2[gaze_index] = row_label_list_1[gaze_index]
            #     if distance_list_1[gaze_index] < distance_threshold and point_pair_list_1[gaze_index][1] != point_pair_list_2[gaze_index][1]:
            #         point_pair_list_2.append(point_pair_list_1[gaze_index])
            #         weight_list_2.append(weight_list_1[gaze_index])
            #         row_label_list_2.append(row_label_list_1[gaze_index])

            for gaze_index in range(len(point_pair_list_2)):
                text_x = point_pair_list_2[gaze_index][1][0]
                text_y = point_pair_list_2[gaze_index][1][1]
                if (text_x, text_y) in actual_text_point_dict:
                    actual_text_point_dict[(text_x, text_y)] += 1
                if (text_x, text_y) in actual_supplement_text_point_dict:
                    actual_supplement_text_point_dict[(text_x, text_y)] += 1

            point_pair_list.extend(point_pair_list_2)
            weight_list.extend(weight_list_2)
            row_label_list.extend(row_label_list_2)

    # 生成一个所有reading point的nearest neighbor。
    total_reading_nbrs = NearestNeighbors(n_neighbors=int(len(gaze_point_list_1d)/4), algorithm='kd_tree').fit(gaze_point_list_1d)
    # 生成每个文本每个reading label的nearest neighbor。
    reading_nbrs_list = []
    for text_index in range(len(reading_data)):
        reading_df = reading_data[text_index]
        reading_coordinates = reading_df[["gaze_x", "gaze_y"]].values.tolist()
        reading_nbrs = NearestNeighbors(n_neighbors=int(len(reading_coordinates)/4), algorithm='kd_tree').fit(reading_coordinates)
        reading_nbrs_list.append(reading_nbrs)

    # 然后找出那些没有任何匹配的actual text point，将其与最近的阅读点匹配。
    total_effective_text_point_num = sum(effective_text_point_dict.values())
    point_pair_length = len(point_pair_list)

    # iterate over actual_text_point_dict
    for key, value in actual_text_point_dict.items():
        if value == 0:
            closet_point_num = int(point_pair_length * effective_text_point_dict[key] / total_effective_text_point_num)
            cur_text_point = [float(key[0]), float(key[1])]
            distances, indices = total_reading_nbrs.kneighbors([cur_text_point])
            # 对于右下角的未被匹配的文本点，我们将其权重放大10倍。
            if (key[0] == configs.right_down_text_center[0] and (key[1] == configs.right_down_text_center[1] or key[1] == configs.right_down_text_center[1] - configs.text_width)) or \
                    (key[0] == configs.right_down_text_center[0] - configs.text_height and key[1] == configs.right_down_text_center[1]):
                weight = configs.completion_weight * configs.right_down_corner_unmatched_ratio
            else:
                weight = configs.completion_weight
            for point_index in range(closet_point_num):
                current_point_index = indices[0][point_index]
                gaze_point = gaze_point_list_1d[current_point_index].tolist()
                point_pair_list.append([gaze_point, cur_text_point])
                weight_list.append(weight)
                row_label_list.append(-1)

    # 对于最左侧的点和最右侧，都可以考虑额外添加一些匹配点对，然后添加的weight是负数。这里最左侧指的是最靠右的补充点或者空格点，最右侧指的是最靠左的不补充点。
    for text_index in range(len(text_data)):
        text_df = text_data[text_index]
        row_list = text_df["row"].unique().tolist()

        for row_index in range(len(row_list)):
            row_df = text_df[text_df["row"] == row_list[row_index]]
            if row_df[row_df["word"] != "blank_supplement"].shape[0] == 0:
                continue

            row_df = row_df.sort_values(by=["col"])
            for index in range(row_df.shape[0]):
                if index < row_df.shape[0] - 1:
                    word = row_df.iloc[index]["word"]
                    next_word = row_df.iloc[index + 1]["word"]
                    if (word == "blank_supplement" or word.strip() == "") and (next_word != "blank_supplement" and next_word.strip() != ""):
                        x = row_df.iloc[index]["x"]
                        y = row_df.iloc[index]["y"]
                        distances, indices = reading_nbrs_list[text_index].kneighbors([[x, y]])
                        for point_index in range(len(indices[0])):
                            if distances[0][point_index] < distance_threshold * configs.left_boundary_distance_threshold_ratio:
                                gaze_point = reading_data[text_index].iloc[indices[0][point_index]][["gaze_x", "gaze_y"]].values.tolist()
                                point_pair_list.append([gaze_point, [x, y]])
                                weight_list.append(configs.empty_penalty * configs.left_boundary_ratio)
                                row_label_list.append(-1)
                            else:
                                break

            row_df = row_df.sort_values(by=["col"], ascending=False)
            for index in range(row_df.shape[0]):
                if index < row_df.shape[0] - 1:
                    word = row_df.iloc[index]["word"]
                    next_word = row_df.iloc[index + 1]["word"]
                    if (word == "blank_supplement" or word.strip() == "") and (next_word != "blank_supplement" and next_word.strip() != ""):
                        x = row_df.iloc[index]["x"]
                        y = row_df.iloc[index]["y"]
                        distances, indices = reading_nbrs_list[text_index].kneighbors([[x, y]])
                        for point_index in range(len(indices[0])):
                            if distances[0][point_index] < distance_threshold * configs.right_boundary_distance_threshold_ratio:
                                gaze_point = reading_data[text_index].iloc[indices[0][point_index]][["gaze_x", "gaze_y"]].values.tolist()
                                point_pair_list.append([gaze_point, [x, y]])
                                weight_list.append(configs.empty_penalty * configs.right_boundary_ratio)
                                row_label_list.append(-1)
                            else:
                                break

        # for index, row in text_df.iterrows():
        #     x = row["x"]
        #     y = row["y"]
        #
        #     distances, indices = total_reading_nbrs.kneighbors([[x, y]])
        #     for point_index in range(len(indices[0])):
        #         if distances[0][point_index] < distance_threshold * configs.boundary_distance_threshold_ratio:
        #             gaze_point = gaze_point_list_1d[indices[0][point_index]].tolist()
        #             point_pair_list.append([gaze_point, [x, y]])
        #             weight_list.append(configs.empty_penalty * configs.boundary_ratio)
        #             row_label_list.append(-1)
        #         else:
        #             break

    return point_pair_list, weight_list, row_label_list


def point_matching_3(reading_data, gaze_point_list_1d, text_data, filtered_text_data_list,
                     total_nbrs_list, row_nbrs_list,
                     effective_text_point_dict, actual_text_point_dict, actual_supplement_text_point_dict,
                     distance_threshold):
    # 首先遍历所有的reading point，找到与其label一致的text point，然后将这些匹配点加入到point_pair_list中。
    point_pair_list = []
    weight_list = []
    row_label_list = []
    for text_index in range(len(reading_data)):
        if text_index in configs.training_index_list:
            continue
        reading_df = reading_data[text_index]
        for row_index in range(configs.row_num):
            filtered_reading_df = reading_df[reading_df["row_label"] == row_index]

            point_pair_list_1 = []
            weight_list_1 = []
            row_label_list_1 = []
            distance_list_1 = []
            if row_nbrs_list[text_index][row_index] and filtered_reading_df.shape[0] != 0:
                filtered_reading_coordinates = filtered_reading_df[["gaze_x", "gaze_y"]].values.tolist()
                filtered_distances_of_row, filtered_indices_of_row = row_nbrs_list[text_index][row_index].kneighbors(filtered_reading_coordinates)
                filtered_text_data = filtered_text_data_list[text_index][row_index]
                filtered_text_coordinate = filtered_text_data[["x", "y"]].values.tolist()
                filtered_reading_density = filtered_reading_df["density"].values.tolist()

                text_coordinate = text_data[text_index][["x", "y"]].values.tolist()
                filtered_distance_of_all_text, filtered_indices_of_all_text = total_nbrs_list[text_index].kneighbors(filtered_reading_coordinates)

                for gaze_index in range(len(filtered_distances_of_row)):
                    if filtered_distances_of_row[gaze_index][0] < distance_threshold:
                        point_pair_list_1.append([filtered_reading_coordinates[gaze_index], filtered_text_coordinate[filtered_indices_of_row[gaze_index][0]]])
                        prediction = filtered_text_data.iloc[filtered_indices_of_row[gaze_index][0]]["prediction"]
                        density = filtered_reading_density[gaze_index]
                        distance_list_1.append(filtered_distances_of_row[gaze_index][0])
                        # weight = 1 / abs(density - prediction) * 5
                        weight = 1 / (abs(density - prediction) / configs.weight_divisor + configs.weight_intercept)
                        if configs.bool_weight:
                            weight_list_1.append(weight)
                        else:
                            weight_list_1.append(1)
                    else:
                        point_pair_list_1.append([filtered_reading_coordinates[gaze_index], text_coordinate[filtered_indices_of_all_text[gaze_index][0]]])
                        if text_data[text_index].iloc[filtered_indices_of_all_text[gaze_index][0]]["word"] == "blank_supplement":
                            weight = text_data[text_index].iloc[filtered_indices_of_all_text[gaze_index][0]]["penalty"]
                        else:
                            prediction = text_data[text_index].iloc[filtered_indices_of_all_text[gaze_index][0]]["prediction"]
                            density = filtered_reading_density[gaze_index]
                            distance_list_1.append(filtered_distance_of_all_text[gaze_index][0])
                            # weight = 1 / abs(density - prediction) * 5
                            weight = 1 / (abs(density - prediction) / configs.weight_divisor + configs.weight_intercept)
                        if configs.bool_weight:
                            weight_list_1.append(weight)
                        else:
                            weight_list_1.append(1)
                    row_label_list_1.append(row_index)

            for gaze_index in range(len(point_pair_list_1)):
                text_x = point_pair_list_1[gaze_index][1][0]
                text_y = point_pair_list_1[gaze_index][1][1]
                if (text_x, text_y) in actual_text_point_dict:
                    actual_text_point_dict[(text_x, text_y)] += 1
                if (text_x, text_y) in actual_supplement_text_point_dict:
                    actual_supplement_text_point_dict[(text_x, text_y)] += 1

            point_pair_list.extend(point_pair_list_1)
            weight_list.extend(weight_list_1)
            row_label_list.extend(row_label_list_1)

    # 生成一个所有reading point的nearest neighbor。
    total_reading_nbrs = NearestNeighbors(n_neighbors=int(len(gaze_point_list_1d)/4), algorithm='kd_tree').fit(gaze_point_list_1d)
    # 生成每个文本每个reading label的nearest neighbor。
    reading_nbrs_list = []
    for text_index in range(len(reading_data)):
        reading_df = reading_data[text_index]
        reading_coordinates = reading_df[["gaze_x", "gaze_y"]].values.tolist()
        reading_nbrs = NearestNeighbors(n_neighbors=int(len(reading_coordinates)/4), algorithm='kd_tree').fit(reading_coordinates)
        reading_nbrs_list.append(reading_nbrs)

    supplement_point_pair_list = []
    supplement_weight_list = []
    supplement_row_label_list = []
    # 然后找出那些没有任何匹配的actual text point，将其与最近的阅读点匹配。
    total_effective_text_point_num = sum(effective_text_point_dict.values())
    point_pair_length = len(point_pair_list)
    # iterate over actual_text_point_dict
    for key, value in actual_text_point_dict.items():
        if value == 0:
            closet_point_num = int(point_pair_length * effective_text_point_dict[key] / total_effective_text_point_num)
            cur_text_point = [float(key[0]), float(key[1])]
            distances, indices = total_reading_nbrs.kneighbors([cur_text_point])
            # 对于右下角的未被匹配的文本点，我们将其权重放大10倍。
            if (key[0] == configs.right_down_text_center[0] and (key[1] == configs.right_down_text_center[1] or key[1] == configs.right_down_text_center[1] - configs.text_width)) or \
                    (key[0] == configs.right_down_text_center[0] - configs.text_height and key[1] == configs.right_down_text_center[1]):
                weight = configs.completion_weight * configs.right_down_corner_unmatched_ratio
            else:
                weight = configs.completion_weight

            for point_index in range(closet_point_num):
                current_point_index = indices[0][point_index]
                gaze_point = gaze_point_list_1d[current_point_index].tolist()
                # point_pair_list.append([gaze_point, cur_text_point])
                # weight_list.append(weight)
                # row_label_list.append(-1)
                supplement_point_pair_list.append([gaze_point, cur_text_point])
                supplement_weight_list.append(weight)
                supplement_row_label_list.append(-1)

    raw_point_pair_length = len(point_pair_list)

    # 这里单独为要添加boundary的point pair生成list，方便后续筛选。
    left_point_pair_list = []
    right_point_pair_list = []
    left_weight_list = []
    right_weight_list = []
    left_row_label_list = []
    right_row_label_list = []

    # 对于最左侧的点和最右侧，都可以考虑额外添加一些匹配点对，然后添加的weight是负数。这里最左侧指的是最靠右的补充点或者空格点，最右侧指的是最靠左的不补充点。
    for text_index in range(len(text_data)):
        text_df = text_data[text_index]
        row_list = text_df["row"].unique().tolist()

        for row_index in range(len(row_list)):
            # 对于最后一行，我们为这里的blank_supplement添加匹配。 # 这里给最后一行添加匹配的效果不好，所以去掉了。
            if row_list[row_index] == 5.5:
                pass
                # row_df = text_df[text_df["row"] == row_list[row_index]]
                # for index in range(row_df.shape[0]):
                #     x = row_df.iloc[index]["x"]
                #     y = row_df.iloc[index]["y"]
                #     distances, indices = reading_nbrs_list[text_index].kneighbors([[x, y]])
                #     for point_index in range(len(indices[0])):
                #         if distances[0][point_index] < distance_threshold * configs.bottom_boundary_distance_threshold_ratio:
                #             gaze_point = reading_data[text_index].iloc[indices[0][point_index]][["gaze_x", "gaze_y"]].values.tolist()
                #             point_pair_list.append([gaze_point, [x, y]])
                #             weight_list.append(configs.empty_penalty * configs.bottom_boundary_ratio)
                #             row_label_list.append(-1)
                #         else:
                #             break
            # 对于其它行，对左右两侧的blank_supplement添加匹配。
            else:
                row_df = text_df[text_df["row"] == row_list[row_index]]
                if row_df[row_df["word"] != "blank_supplement"].shape[0] == 0:
                    continue

                row_df = row_df.sort_values(by=["col"])
                for index in range(row_df.shape[0]):
                    if index < row_df.shape[0] - 1:
                        word = row_df.iloc[index]["word"]
                        next_word = row_df.iloc[index + 1]["word"]
                        if (word == "blank_supplement" or word.strip() == "") and (next_word != "blank_supplement" and next_word.strip() != ""):
                            x = row_df.iloc[index]["x"]
                            y = row_df.iloc[index]["y"]
                            distances, indices = reading_nbrs_list[text_index].kneighbors([[x, y]])
                            for point_index in range(len(indices[0])):
                                if distances[0][point_index] < distance_threshold * configs.left_boundary_distance_threshold_ratio:
                                    gaze_point = reading_data[text_index].iloc[indices[0][point_index]][["gaze_x", "gaze_y"]].values.tolist()
                                    # point_pair_list.append([gaze_point, [x, y]])
                                    # weight_list.append(configs.empty_penalty * configs.left_boundary_ratio)
                                    # row_label_list.append(-1)
                                    left_point_pair_list.append([gaze_point, [x, y]])
                                    left_weight_list.append(configs.empty_penalty * configs.left_boundary_ratio)
                                    left_row_label_list.append(-1)
                                else:
                                    break
                            # 确保只对最左侧的空格点添加一次匹配。
                            break

                row_df = row_df.sort_values(by=["col"], ascending=False)
                for index in range(row_df.shape[0]):
                    if index < row_df.shape[0] - 1:
                        word = row_df.iloc[index]["word"]
                        next_word = row_df.iloc[index + 1]["word"]
                        if (word == "blank_supplement" or word.strip() == "") and (next_word != "blank_supplement" and next_word.strip() != ""):
                            x = row_df.iloc[index]["x"]
                            y = row_df.iloc[index]["y"]
                            distances, indices = reading_nbrs_list[text_index].kneighbors([[x, y]])
                            for point_index in range(len(indices[0])):
                                if distances[0][point_index] < distance_threshold * configs.right_boundary_distance_threshold_ratio:
                                    gaze_point = reading_data[text_index].iloc[indices[0][point_index]][["gaze_x", "gaze_y"]].values.tolist()
                                    # point_pair_list.append([gaze_point, [x, y]])
                                    # weight_list.append(configs.empty_penalty * configs.right_boundary_ratio)
                                    # row_label_list.append(-1)
                                    right_point_pair_list.append([gaze_point, [x, y]])
                                    right_weight_list.append(configs.empty_penalty * configs.right_boundary_ratio)
                                    right_row_label_list.append(-1)
                                else:
                                    break
                            # 确保只对最右侧的空格点添加一次匹配。
                            break

    # 对于那些选择部分文本匹配的点，需要限制supplement point pair和raw point pair的比例，避免出现过分的失调。
    if len(configs.training_index_list) > 16:
        supplement_select_indices = np.random.choice(len(supplement_point_pair_list), int(len(supplement_point_pair_list) * configs.supplement_select_ratio), replace=False)
        supplement_point_pair_list = [supplement_point_pair_list[i] for i in supplement_select_indices]
        supplement_weight_list = [supplement_weight_list[i] for i in supplement_select_indices]
        supplement_row_label_list = [supplement_row_label_list[i] for i in supplement_select_indices]
    point_pair_list.extend(supplement_point_pair_list)
    weight_list.extend(supplement_weight_list)
    row_label_list.extend(supplement_row_label_list)

    # 对于那些选择部分文本匹配的点，需要限制boundary point pair和raw point pair的比例，避免出现过分的失调。
    if len(configs.training_index_list) > 16:
        left_select_indices = np.random.choice(len(left_point_pair_list), int(len(left_point_pair_list) * configs.boundary_select_ratio), replace=False)
        left_point_pair_list = [left_point_pair_list[i] for i in left_select_indices]
        left_weight_list = [left_weight_list[i] for i in left_select_indices]
        left_row_label_list = [left_row_label_list[i] for i in left_select_indices]
        right_select_indices = np.random.choice(len(right_point_pair_list), int(len(right_point_pair_list) * configs.boundary_select_ratio), replace=False)
        right_point_pair_list = [right_point_pair_list[i] for i in right_select_indices]
        right_weight_list = [right_weight_list[i] for i in right_select_indices]
        right_row_label_list = [right_row_label_list[i] for i in right_select_indices]
    point_pair_list.extend(left_point_pair_list)
    weight_list.extend(left_weight_list)
    row_label_list.extend(left_row_label_list)
    point_pair_list.extend(right_point_pair_list)
    weight_list.extend(right_weight_list)
    row_label_list.extend(right_row_label_list)

    return point_pair_list, weight_list, row_label_list


def rotating_calipers(points):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    min_area_rect = None
    min_area = float('inf')

    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]

        # 计算边界向量
        edge = p2 - p1
        edge_angle = atan2(edge[1], edge[0])
        if abs(edge_angle) > (np.pi / 4):
            continue
        # 旋转点以使边界水平
        rot_matrix = np.array([[np.cos(edge_angle), np.sin(edge_angle)],
                               [-np.sin(edge_angle), np.cos(edge_angle)]])
        rotated_points = np.dot(rot_matrix, hull_points.T).T

        # 计算旋转后点的边界
        min_x, max_x = np.min(rotated_points[:, 0]), np.max(rotated_points[:, 0])
        min_y, max_y = np.min(rotated_points[:, 1]), np.max(rotated_points[:, 1])

        # 计算矩形的面积
        area = (max_x - min_x) * (max_y - min_y)

        if area < min_area:
            min_area = area
            min_area_rect = (min_x, max_x, min_y, max_y, -edge_angle)

    return min_area_rect


def calibrate_with_location_coverage_penalty_and_rowlabel(subject_index, reading_data, text_data, calibration_data, max_iteration=100, distance_threshold=64):
    reading_data = reading_data.copy()

    # 获取1d的gaze point list。
    total_gaze_point_num = 0
    gaze_point_list_1d = []
    for text_index in range(len(reading_data)):
        coordinates = reading_data[text_index][["gaze_x", "gaze_y"]].values.tolist()
        gaze_point_list_1d.extend(coordinates)
        total_gaze_point_num += len(coordinates)
    gaze_point_list_1d = np.array(gaze_point_list_1d)

    # 用一个dict来记录所有有效的文本点。
    text_point_dict = {}
    for row_index in range(len(calibration_data[subject_index][2])):
        for col_index in range(len(calibration_data[subject_index][2][row_index])):
            x = calibration_data[subject_index][2][row_index][col_index]["point_x"]
            y = calibration_data[subject_index][2][row_index][col_index]["point_y"]
            text_point_dict[(x, y)] = 0
    for text_index in range(len(text_data)):
        for index, row in text_data[text_index].iterrows():
            x = row["x"]
            y = row["y"]
            word = row["word"]
            if word != "blank_supplement" and (x, y) in text_point_dict:
                text_point_dict[(x, y)] += 1
    # 用一个dict来记录非blank_supplement，且至少有过一次文字的文本点。
    effective_text_point_dict = {}
    for key in text_point_dict:
        if text_point_dict[key] != 0:
            effective_text_point_dict[key] = text_point_dict[key]
    text_point_total_utilized_count = 0
    for key in effective_text_point_dict:
        text_point_total_utilized_count += effective_text_point_dict[key]
    # 用一个dict来记录blank_supplement的文本点。
    supplement_text_point_dict = {}
    for text_index in range(len(text_data)):
        for index, row in text_data[text_index].iterrows():
            x = row["x"]
            y = row["y"]
            word = row["word"]
            if word == "blank_supplement":
                supplement_text_point_dict[(x, y)] = 0

    # 按文本、行号来对text point分类，然后据此生成对应的nearest neighbor。
    row_nbrs_list = [[] for _ in range(len(text_data))]
    filtered_text_data_list = [[] for _ in range(len(text_data))]
    for text_index in range(len(text_data)):
        for row_index in range(configs.row_num):
            df = text_data[text_index]
            filtered_text_data_df = df[(df["row"] == float(row_index)) & (df["word"] != "blank_supplement")]
            filtered_text_data_list[text_index].append(filtered_text_data_df)
            if filtered_text_data_df.shape[0] > 0:
                filtered_text_coordinate = filtered_text_data_df[["x", "y"]].values.tolist()
                nbr = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(filtered_text_coordinate)
                row_nbrs_list[text_index].append(nbr)
            else:
                row_nbrs_list[text_index].append(None)

    # 生成一个所有text_point的nearest neighbor。
    total_nbrs_list = []
    for text_index in range(len(text_data)):
        df = text_data[text_index]
        text_coordinate = df[["x", "y"]].values.tolist()
        nbr = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(text_coordinate)
        total_nbrs_list.append(nbr)

    # 将gaze data和text point先做缩放，然后基于重心对齐。
    # fig = plt.figure(figsize=(24, 12))
    # ax = fig.add_subplot(111)
    # ax.set_xlim(0, 1920)
    # ax.set_ylim(800, 0)
    # ax.set_aspect("equal")
    # ax.scatter(gaze_point_list_1d[:, 0], gaze_point_list_1d[:, 1], c='orange', marker='o', s=1, zorder=1)

    dbscan = DBSCAN(eps=32, min_samples=5)
    clusters = dbscan.fit_predict(gaze_point_list_1d)
    filtered_gaze_point_list_1d = gaze_point_list_1d[clusters != -1]

    # ax.scatter(filtered_gaze_point_list_1d[:, 0], filtered_gaze_point_list_1d[:, 1], c='b', marker='o', s=1, zorder=1)

    outer_rect = rotating_calipers(filtered_gaze_point_list_1d)
    x_scale = (configs.right_down_text_center[0] - configs.left_top_text_center[0]) / (outer_rect[1] - outer_rect[0])
    y_scale = (configs.right_down_text_center[1] - configs.left_top_text_center[1]) / (outer_rect[3] - outer_rect[2])
    scale_matrix = np.array([[x_scale, 0, 0],
                             [0, y_scale, 0],
                             [0, 0, 1]])

    gaze_point_list_1d_homogeneous = [UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_point) for gaze_point in gaze_point_list_1d]
    gaze_point_list_1d_homogeneous = [np.dot(scale_matrix, gaze_point) for gaze_point in gaze_point_list_1d_homogeneous]
    gaze_point_list_1d = np.array([UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_point_list_1d_homogeneous])

    filtered_gaze_point_max_x = np.max(gaze_point_list_1d[clusters != -1][:, 0])
    filtered_gaze_point_min_x = np.min(gaze_point_list_1d[clusters != -1][:, 0])
    filtered_gaze_point_max_y = np.max(gaze_point_list_1d[clusters != -1][:, 1])
    filtered_gaze_point_min_y = np.min(gaze_point_list_1d[clusters != -1][:, 1])
    gaze_point_center = np.array([(filtered_gaze_point_max_x + filtered_gaze_point_min_x) / 2,
                                  (filtered_gaze_point_max_y + filtered_gaze_point_min_y) / 2])
    text_point_center = np.array([0, 0])
    for key, value in effective_text_point_dict.items():
        text_point_center[0] += key[0]
        text_point_center[1] += key[1]
    text_point_center[0] /= len(effective_text_point_dict)
    text_point_center[1] /= len(effective_text_point_dict)
    translate_vector = np.array(text_point_center - gaze_point_center)
    translate_matrix = np.array([[1, 0, translate_vector[0]],
                                 [0, 1, translate_vector[1]],
                                 [0, 0, 1]])

    gaze_point_list_1d_homogeneous = [UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_point) for gaze_point in gaze_point_list_1d]
    gaze_point_list_1d_homogeneous = [np.dot(translate_matrix, gaze_point) for gaze_point in gaze_point_list_1d_homogeneous]
    gaze_point_list_1d = np.array([UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_point_list_1d_homogeneous])

    # ax.scatter(gaze_point_list_1d[:, 0], gaze_point_list_1d[:, 1], c='g', marker='o', s=1, zorder=1)
    # plt.show()

    for text_index in range(len(reading_data)):
        gaze_x = reading_data[text_index]["gaze_x"]
        gaze_y = reading_data[text_index]["gaze_y"]
        gaze_homogeneous = [UtilFunctions.change_2d_vector_to_homogeneous_vector([gaze_x[i], gaze_y[i]]) for i in range(len(gaze_x))]
        gaze_homogeneous = [np.dot(translate_matrix, np.dot(scale_matrix, gaze_point)) for gaze_point in gaze_homogeneous]
        gaze_1d = [UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_homogeneous]
        reading_data[text_index]["gaze_x"] = [gaze_1d[i][0] for i in range(len(gaze_1d))]
        reading_data[text_index]["gaze_y"] = [gaze_1d[i][1] for i in range(len(gaze_1d))]

    # 把calibration的数据也做一下相同的变换。
    for row_index in range(len(calibration_data[subject_index][1])):
        for col_index in range(len(calibration_data[subject_index][1][row_index])):
            avg_gaze_x = calibration_data[subject_index][1][row_index][col_index]["avg_gaze_x"]
            avg_gaze_y = calibration_data[subject_index][1][row_index][col_index]["avg_gaze_y"]
            avg_gaze_homogeneous = UtilFunctions.change_2d_vector_to_homogeneous_vector([avg_gaze_x, avg_gaze_y])
            avg_gaze_homogeneous = np.dot(translate_matrix, np.dot(scale_matrix, avg_gaze_homogeneous))
            avg_gaze_2d = UtilFunctions.change_homogeneous_vector_to_2d_vector(avg_gaze_homogeneous)
            calibration_data[subject_index][1][row_index][col_index]["avg_gaze_x"] = avg_gaze_2d[0]
            calibration_data[subject_index][1][row_index][col_index]["avg_gaze_y"] = avg_gaze_2d[1]

            gaze_x = calibration_data[subject_index][0][row_index][col_index]["gaze_x"]
            gaze_y = calibration_data[subject_index][0][row_index][col_index]["gaze_y"]
            gaze_homogeneous = [UtilFunctions.change_2d_vector_to_homogeneous_vector([gaze_x[i], gaze_y[i]]) for i in range(len(gaze_x))]
            gaze_homogeneous = [np.dot(translate_matrix, np.dot(scale_matrix, gaze_point)) for gaze_point in gaze_homogeneous]
            gaze_2d = [UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_homogeneous]
            calibration_data[subject_index][0][row_index][col_index]["gaze_x"] = [gaze_2d[i][0] for i in range(len(gaze_2d))]
            calibration_data[subject_index][0][row_index][col_index]["gaze_y"] = [gaze_2d[i][1] for i in range(len(gaze_2d))]

    total_transform_matrix = np.eye(3)
    avg_error_list = []
    last_iteration_num_list = []
    last_iteration_num = 100000
    gd_error_list = []
    # with multiprocessing.Pool(processes=configs.number_of_process) as pool:
    for iteration_index in range(max_iteration):
        print("iteration_index: ", iteration_index)
        # 每次迭代前，创建一个类似effective_text_point_dict的字典，用于记录每个文本点被阅读点覆盖的次数。
        actual_text_point_dict = effective_text_point_dict.copy()
        for key in actual_text_point_dict:
            actual_text_point_dict[key] = 0

        actual_supplement_text_point_dict = supplement_text_point_dict.copy()

        '''
        # 这里的多线程是负优化。
        args_list = []
        for text_index in range(len(reading_data)):
            for row_index in range(configs.row_num):
                args_list.append((reading_data, text_data,
                                  text_index, row_index,
                                  filtered_text_data_list,
                                  total_nbrs_list, row_nbrs_list,
                                  actual_text_point_dict, actual_supplement_text_point_dict,
                                  distance_threshold))

        result_list = pool.starmap(single_process_matching, args_list)
        result_list = np.array(result_list)
        point_pair_list = result_list[:, 0]
        point_pair_list = [point_pair for point_pair in point_pair_list if len(point_pair) > 0]
        point_pair_list = np.concatenate(point_pair_list, axis=0)
        weight_list = result_list[:, 1]
        weight_list = [weight for weight in weight_list if len(weight) > 0]
        weight_list = np.concatenate(weight_list, axis=0)
        row_label_list = result_list[:, 2]
        row_label_list = [row_label for row_label in row_label_list if len(row_label) > 0]
        row_label_list = np.concatenate(row_label_list, axis=0)
        '''
        #
        # point_pair_list, weight_list, row_label_list = point_matching_1(reading_data, text_data, filtered_text_data_list,
        #                                                                 total_nbrs_list, row_nbrs_list,
        #                                                                 actual_text_point_dict, actual_supplement_text_point_dict,
        #                                                                 distance_threshold)

        point_pair_list, weight_list, row_label_list = point_matching_2(reading_data, gaze_point_list_1d, text_data, filtered_text_data_list,
                                                                        total_nbrs_list, row_nbrs_list,
                                                                        effective_text_point_dict, actual_text_point_dict, actual_supplement_text_point_dict,
                                                                        distance_threshold)
        # print(f"first 10 point pairs: {point_pair_list[:10]}\n"
        #       f"last 10 point pairs: {point_pair_list[-10:]}\n")

        # fig = plt.figure(figsize=(24, 12))
        # ax = fig.add_subplot(111)
        # ax.set_xlim(0, 1920)
        # ax.set_ylim(800, 0)
        # ax.set_aspect("equal")
        # color_list = [(0.5, 0.5, 0), (0, 1, 0), (0.5, 0.5, 1), (1, 0.5, 0.5), (0.5, 1, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0.5)]
        # ax.scatter(np.array(point_pair_list)[:, 0, 0], np.array(point_pair_list)[:, 0, 1], c=[color_list[i] for i in row_label_list], marker='o', s=1, zorder=1)
        # ax.scatter(np.array(point_pair_list)[:, 1, 0], np.array(point_pair_list)[:, 1, 1], c=[color_list[i] for i in row_label_list], marker='x', s=10, zorder=1)
        # line_segment_list = []
        # color_list = []
        # for point_pair_index in range(len(point_pair_list)):
        #     line_segment_list.append([point_pair_list[point_pair_index][0], point_pair_list[point_pair_index][1]])
        #     if weight_list[point_pair_index] > 0:
        #         color_list.append("b")
        #     else:
        #         color_list.append("r")
        # line_collection = LineCollection(line_segment_list, colors=color_list, linewidths=0.5, zorder=0)
        # ax.add_collection(line_collection)
        # plt.show()

        transform_matrix, gd_error, last_iteration_num = GradientDescent.gradient_descent_with_whole_matrix_using_tensor_with_weight(point_pair_list, weight_list, last_iteration_num=last_iteration_num, max_iterations=3000)

        gd_error_list.append(gd_error)
        # print(f"transform_matrix: {transform_matrix}")
        # update total_transform_matrix
        total_transform_matrix = np.dot(transform_matrix, total_transform_matrix)

        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
            avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
            calibration_point_list_modified = ManualCalibrateForStd.apply_transform_to_calibration(subject_index, calibration_data, total_transform_matrix)

        fig = plt.figure(figsize=(24, 12))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1920)
        ax.set_ylim(800, 0)
        ax.set_aspect("equal")

        # 将移动前的gaze_point用橙色标记。
        ax.scatter(gaze_point_list_1d[:, 0], gaze_point_list_1d[:, 1], c='orange', marker='o', s=1, zorder=1)

        # update gaze_point_list_1d
        gaze_point_list_1d = [UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_point) for gaze_point in gaze_point_list_1d]
        gaze_point_list_1d = [np.dot(transform_matrix, gaze_point) for gaze_point in gaze_point_list_1d]
        gaze_point_list_1d = np.array([UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_point_list_1d])
        # update reading_data
        for text_index in range(len(reading_data)):
            gaze_coordinates = reading_data[text_index][["gaze_x", "gaze_y"]].values.tolist()
            gaze_coordinates = [UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates = [np.dot(transform_matrix, gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates = [UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            reading_data[text_index][["gaze_x", "gaze_y"]] = gaze_coordinates

        # 将移动后的gaze_point用绿色标记。
        ax.scatter(gaze_point_list_1d[:, 0], gaze_point_list_1d[:, 1], c='g', marker='o', s=1, zorder=1)

        # TODO 这里先简单写一个看效果的demo，之后再将函数做合适的封装处理。
        distance_list, avg_distance = ManualCalibrateForStd.compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        avg_error_list.append(avg_distance)
        print(f"average distance: {avg_distance}, last iteration num: {last_iteration_num}")
        last_iteration_num_list.append(last_iteration_num)

        max_pair_num = max(actual_text_point_dict.values())
        for key, value in actual_text_point_dict.items():
            color_ratio = 0.8 - (value / max_pair_num) * 0.6
            color = (color_ratio, color_ratio, color_ratio)
            if value == 0:
                ax.scatter(key[0], key[1], c=color, marker='x', s=40, zorder=2)
            else:
                ax.scatter(key[0], key[1], c=color, marker='o', s=40, zorder=2)

        max_pair_num = max(actual_supplement_text_point_dict.values())
        for key, value in actual_supplement_text_point_dict.items():
            color_ratio = 0.8 - (value / max_pair_num) * 0.6
            color = (1, color_ratio, color_ratio)
            if value == 0:
                ax.scatter(key[0], key[1], c=color, marker='x', s=10, zorder=3)
            else:
                ax.scatter(key[0], key[1], c=color, marker='o', s=10, zorder=3)

        color_list = []
        line_segment_list = []
        for point_pair_index in range(len(point_pair_list)):
            # point_pair_gaze = UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair_list[point_pair_index][0])
            # point_pair_gaze = np.dot(transform_matrix, point_pair_gaze)
            # point_pair_gaze = UtilFunctions.change_homogeneous_vector_to_2d_vector(point_pair_gaze)
            if weight_list[point_pair_index] > 0:
                color_list.append("b")
            else:
                color_list.append("r")
            # point_pair_gaze = last_point_pair_list[point_pair_index][0]
            line_segment_list.append([point_pair_list[point_pair_index][0], point_pair_list[point_pair_index][1]])
        line_collection = LineCollection(line_segment_list, colors=color_list, linewidths=0.5, zorder=0)
        ax.add_collection(line_collection)
        gaze_file_path = f"pic/reading_matching/gaze_matching/subject_{subject_index}"
        if not os.path.exists(gaze_file_path):
            os.makedirs(gaze_file_path)
        plt.savefig(f"{gaze_file_path}/iteration_{iteration_index}.png")
        # plt.show()

        calibration_file_path = f"pic/reading_matching/calibration/subject_{subject_index}"
        if not os.path.exists(calibration_file_path):
            os.makedirs(calibration_file_path)
        Render.visualize_cali_result(gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list,
                                     avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list,
                                     calibration_point_list_modified, file_name=f"{calibration_file_path}/iteration_{iteration_index}.png")

    # log_path = "log/gradient_descent_avg_error"
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path)
    # log_file = open(f"{log_path}/subject_{subject_index}.txt", "w")
    # log_file.write(f"location_penalty: {configs.location_penalty}, punctuation_penalty: {configs.punctuation_penalty}, "
    #                f"empty_penalty: {configs.empty_penalty}, completion_weight: {configs.completion_weight},"
    #                f"right_down_corner_un_matched_ratio: {configs.right_down_corner_un_matched_ratio}, left_boundary_ratio: {configs.left_boundary_ratio},"
    #                f"left_boundary_distance_threshold_ratio: {configs.left_boundary_distance_threshold_ratio}, gradient_descent_stop_accuracy: {configs.gradient_descent_stop_accuracy}\n")
    #
    # for iteration_index in range(len(avg_error_list)):
    #     print("avg_error_list[", iteration_index, "]: ", avg_error_list[iteration_index])
    #     log_file.write(
    #         f"avg_error_list[{iteration_index}]: {avg_error_list[iteration_index]}, last_iteration_num: {last_iteration_num_list[iteration_index]}, last_gd_error: {gd_error_list[iteration_index]}\n")
    #
    # log_file.close()

    return avg_error_list


def transform_using_centroid_and_outbound(gaze_point_list_1d, effective_text_point_dict, reading_data, subject_index, calibration_data):
    # 将gaze data和text point先做缩放，然后基于重心对齐。
    # fig = plt.figure(figsize=(24, 12))
    # ax = fig.add_subplot(111)
    # ax.set_xlim(0, 1920)
    # ax.set_ylim(800, 0)
    # ax.set_aspect("equal")
    # ax.scatter(gaze_point_list_1d[:, 0], gaze_point_list_1d[:, 1], c='orange', marker='o', s=1, zorder=1)

    dbscan = DBSCAN(eps=32, min_samples=5)
    clusters = dbscan.fit_predict(gaze_point_list_1d)
    filtered_gaze_point_list_1d = gaze_point_list_1d[clusters != -1]

    # ax.scatter(filtered_gaze_point_list_1d[:, 0], filtered_gaze_point_list_1d[:, 1], c='b', marker='o', s=1, zorder=1)

    outer_rect = rotating_calipers(filtered_gaze_point_list_1d)
    x_scale = (configs.right_down_text_center[0] - configs.left_top_text_center[0]) / (outer_rect[1] - outer_rect[0])
    y_scale = (configs.right_down_text_center[1] - configs.left_top_text_center[1]) / (outer_rect[3] - outer_rect[2])
    scale_matrix = np.array([[x_scale, 0, 0],
                             [0, y_scale, 0],
                             [0, 0, 1]])

    gaze_point_list_1d_homogeneous = [UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_point) for gaze_point in gaze_point_list_1d]
    gaze_point_list_1d_homogeneous = [np.dot(scale_matrix, gaze_point) for gaze_point in gaze_point_list_1d_homogeneous]
    gaze_point_list_1d = np.array([UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_point_list_1d_homogeneous])

    filtered_gaze_point_max_x = np.max(gaze_point_list_1d[clusters != -1][:, 0])
    filtered_gaze_point_min_x = np.min(gaze_point_list_1d[clusters != -1][:, 0])
    filtered_gaze_point_max_y = np.max(gaze_point_list_1d[clusters != -1][:, 1])
    filtered_gaze_point_min_y = np.min(gaze_point_list_1d[clusters != -1][:, 1])
    gaze_point_center = np.array([(filtered_gaze_point_max_x + filtered_gaze_point_min_x) / 2,
                                  (filtered_gaze_point_max_y + filtered_gaze_point_min_y) / 2])
    text_point_center = np.array([0, 0])
    for key, value in effective_text_point_dict.items():
        text_point_center[0] += key[0]
        text_point_center[1] += key[1]
    text_point_center[0] /= len(effective_text_point_dict)
    text_point_center[1] /= len(effective_text_point_dict)
    translate_vector = np.array(text_point_center - gaze_point_center)
    translate_matrix = np.array([[1, 0, translate_vector[0]],
                                 [0, 1, translate_vector[1]],
                                 [0, 0, 1]])

    gaze_point_list_1d_homogeneous = [UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_point) for gaze_point in gaze_point_list_1d]
    gaze_point_list_1d_homogeneous = [np.dot(translate_matrix, gaze_point) for gaze_point in gaze_point_list_1d_homogeneous]
    gaze_point_list_1d = np.array([UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_point_list_1d_homogeneous])

    # ax.scatter(gaze_point_list_1d[:, 0], gaze_point_list_1d[:, 1], c='g', marker='o', s=1, zorder=1)
    # plt.show()

    for text_index in range(len(reading_data)):
        gaze_x = reading_data[text_index]["gaze_x"]
        gaze_y = reading_data[text_index]["gaze_y"]
        gaze_homogeneous = [UtilFunctions.change_2d_vector_to_homogeneous_vector([gaze_x[i], gaze_y[i]]) for i in range(len(gaze_x))]
        gaze_homogeneous = [np.dot(translate_matrix, np.dot(scale_matrix, gaze_point)) for gaze_point in gaze_homogeneous]
        gaze_1d = [UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_homogeneous]
        reading_data[text_index]["gaze_x"] = [gaze_1d[i][0] for i in range(len(gaze_1d))]
        reading_data[text_index]["gaze_y"] = [gaze_1d[i][1] for i in range(len(gaze_1d))]

    # 把calibration的数据也做一下相同的变换。
    for row_index in range(len(calibration_data[subject_index][1])):
        for col_index in range(len(calibration_data[subject_index][1][row_index])):
            avg_gaze_x = calibration_data[subject_index][1][row_index][col_index]["avg_gaze_x"]
            avg_gaze_y = calibration_data[subject_index][1][row_index][col_index]["avg_gaze_y"]
            avg_gaze_homogeneous = UtilFunctions.change_2d_vector_to_homogeneous_vector([avg_gaze_x, avg_gaze_y])
            avg_gaze_homogeneous = np.dot(translate_matrix, np.dot(scale_matrix, avg_gaze_homogeneous))
            avg_gaze_2d = UtilFunctions.change_homogeneous_vector_to_2d_vector(avg_gaze_homogeneous)
            calibration_data[subject_index][1][row_index][col_index]["avg_gaze_x"] = avg_gaze_2d[0]
            calibration_data[subject_index][1][row_index][col_index]["avg_gaze_y"] = avg_gaze_2d[1]

            gaze_x = calibration_data[subject_index][0][row_index][col_index]["gaze_x"]
            gaze_y = calibration_data[subject_index][0][row_index][col_index]["gaze_y"]
            gaze_homogeneous = [UtilFunctions.change_2d_vector_to_homogeneous_vector([gaze_x[i], gaze_y[i]]) for i in range(len(gaze_x))]
            gaze_homogeneous = [np.dot(translate_matrix, np.dot(scale_matrix, gaze_point)) for gaze_point in gaze_homogeneous]
            gaze_2d = [UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_homogeneous]
            calibration_data[subject_index][0][row_index][col_index]["gaze_x"] = [gaze_2d[i][0] for i in range(len(gaze_2d))]
            calibration_data[subject_index][0][row_index][col_index]["gaze_y"] = [gaze_2d[i][1] for i in range(len(gaze_2d))]

    return gaze_point_list_1d, effective_text_point_dict, reading_data, calibration_data


def create_text_nearest_neighbor(text_data):
    # 按文本、行号来对text point分类，然后据此生成对应的nearest neighbor。
    row_nbrs_list = [[] for _ in range(len(text_data))]
    for text_index in range(len(text_data)):
        for row_index in range(configs.row_num):
            df = text_data[text_index]
            filtered_text_data_df = df[(df["row"] == float(row_index)) & (df["word"] != "blank_supplement")]
            if filtered_text_data_df.shape[0] > 0:
                filtered_text_coordinate = filtered_text_data_df[["x", "y"]].values.tolist()
                nbr = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(filtered_text_coordinate)
                row_nbrs_list[text_index].append(nbr)
            else:
                row_nbrs_list[text_index].append(None)

    # 生成一个所有text_point的nearest neighbor。
    total_nbrs_list = []
    for text_index in range(len(text_data)):
        df = text_data[text_index]
        text_coordinate = df[["x", "y"]].values.tolist()
        nbr = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(text_coordinate)
        total_nbrs_list.append(nbr)

    return row_nbrs_list, total_nbrs_list


def prepare_dicts_for_text_point(subject_index, text_data, calibration_data):
    # 用一个dict来记录所有有效的文本点。
    text_point_dict = {}
    for row_index in range(len(calibration_data[subject_index][2])):
        for col_index in range(len(calibration_data[subject_index][2][row_index])):
            x = calibration_data[subject_index][2][row_index][col_index]["point_x"]
            y = calibration_data[subject_index][2][row_index][col_index]["point_y"]
            text_point_dict[(x, y)] = 0
    for text_index in range(len(text_data)):
        for index, row in text_data[text_index].iterrows():
            x = row["x"]
            y = row["y"]
            word = row["word"]
            if word != "blank_supplement" and (x, y) in text_point_dict:
                text_point_dict[(x, y)] += 1
    # 用一个dict来记录非blank_supplement，且至少有过一次文字的文本点。
    effective_text_point_dict = {}
    for key in text_point_dict:
        if text_point_dict[key] != 0:
            effective_text_point_dict[key] = text_point_dict[key]
    text_point_total_utilized_count = 0
    for key in effective_text_point_dict:
        text_point_total_utilized_count += effective_text_point_dict[key]
    # 用一个dict来记录blank_supplement的文本点。
    supplement_text_point_dict = {}
    for text_index in range(len(text_data)):
        for index, row in text_data[text_index].iterrows():
            x = row["x"]
            y = row["y"]
            word = row["word"]
            if word == "blank_supplement":
                supplement_text_point_dict[(x, y)] = 0

    return text_point_dict, effective_text_point_dict, supplement_text_point_dict, text_point_total_utilized_count


def calibrate_with_torch(subject_index, reading_data, text_data, calibration_data, max_iteration=100, distance_threshold=64):
    reading_data = reading_data.copy()

    # 获取1d的gaze point list。
    total_gaze_point_num = 0
    gaze_point_list_1d = []
    for text_index in range(len(reading_data)):
        coordinates = reading_data[text_index][["gaze_x", "gaze_y"]].values.tolist()
        gaze_point_list_1d.extend(coordinates)
        total_gaze_point_num += len(coordinates)
    gaze_point_list_1d = np.array(gaze_point_list_1d)

    filtered_text_data_list = [[] for _ in range(len(text_data))]
    for text_index in range(len(text_data)):
        for row_index in range(configs.row_num):
            df = text_data[text_index]
            filtered_text_data_df = df[(df["row"] == float(row_index)) & (df["word"] != "blank_supplement")]
            filtered_text_data_list[text_index].append(filtered_text_data_df)

    # 生成3个dict。text_point_dict用来记录所有有效的文本点；effective_text_point_dict用来记录非blank_supplement，且至少有过一次文字的文本点。supplement_text_point_dict用来记录blank_supplement的文本点。
    text_point_dict, effective_text_point_dict, supplement_text_point_dict, text_point_total_utilized_count = prepare_dicts_for_text_point(subject_index, text_data, calibration_data)
    # 生成：按文本、行号来对text point分类得到的nearest neighbor；按文本、行号来对text point分类，且去除blank_supplement的nearest neighbor；以及所有text_point的nearest neighbor。
    row_nbrs_list, total_nbrs_list = create_text_nearest_neighbor(text_data)
    # 将gaze data和text point先做缩放，然后基于重心对齐。
    gaze_point_list_1d, effective_text_point_dict, reading_data, calibration_data = transform_using_centroid_and_outbound(gaze_point_list_1d, effective_text_point_dict, reading_data, subject_index, calibration_data)

    total_transform_matrix = np.eye(3)
    avg_error_list = []
    last_iteration_num_list = []
    last_iteration_num = 100000
    gd_error_list = []
    learning_rate_list = []
    for iteration_index in range(max_iteration):
        print("iteration_index: ", iteration_index)
        # 每次迭代前，创建一个类似effective_text_point_dict的字典，用于记录每个文本点被阅读点覆盖的次数。
        actual_text_point_dict = effective_text_point_dict.copy()
        for key in actual_text_point_dict:
            actual_text_point_dict[key] = 0

        actual_supplement_text_point_dict = supplement_text_point_dict.copy()
        point_pair_list, weight_list, row_label_list = point_matching_2(reading_data, gaze_point_list_1d, text_data, filtered_text_data_list,
                                                                        total_nbrs_list, row_nbrs_list,
                                                                        effective_text_point_dict, actual_text_point_dict, actual_supplement_text_point_dict,
                                                                        distance_threshold)
        # print(f"first 10 point pairs: {point_pair_list[:10]}\n"
        #       f"last 10 point pairs: {point_pair_list[-10:]}\n")

        # fig = plt.figure(figsize=(24, 12))
        # ax = fig.add_subplot(111)
        # ax.set_xlim(0, 1920)
        # ax.set_ylim(800, 0)
        # ax.set_aspect("equal")
        # color_list = [(0.5, 0.5, 0), (0, 1, 0), (0.5, 0.5, 1), (1, 0.5, 0.5), (0.5, 1, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0.5)]
        # ax.scatter(np.array(point_pair_list)[:, 0, 0], np.array(point_pair_list)[:, 0, 1], c=[color_list[i] for i in row_label_list], marker='o', s=1, zorder=1)
        # ax.scatter(np.array(point_pair_list)[:, 1, 0], np.array(point_pair_list)[:, 1, 1], c=[color_list[i] for i in row_label_list], marker='x', s=10, zorder=1)
        # line_segment_list = []
        # color_list = []
        # for point_pair_index in range(len(point_pair_list)):
        #     line_segment_list.append([point_pair_list[point_pair_index][0], point_pair_list[point_pair_index][1]])
        #     if weight_list[point_pair_index] > 0:
        #         color_list.append("b")
        #     else:
        #         color_list.append("r")
        # line_collection = LineCollection(line_segment_list, colors=color_list, linewidths=0.5, zorder=0)
        # ax.add_collection(line_collection)
        # plt.show()

        learning_rate = 2e-2
        if iteration_index > int(max_iteration / 2):
            learning_rate = 1e-2
        learning_rate_list.append(learning_rate)
        transform_matrix, gd_error, last_iteration_num, _ = GradientDescent.gradient_descent_with_torch(point_pair_list, weight_list, learning_rate=learning_rate, last_iteration_num=last_iteration_num, max_iterations=3000)

        gd_error_list.append(gd_error)
        # print(f"transform_matrix: {transform_matrix}")
        # update total_transform_matrix
        total_transform_matrix = np.dot(transform_matrix, total_transform_matrix)

        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
            avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
            calibration_point_list_modified = ManualCalibrateForStd.apply_transform_to_calibration(subject_index, calibration_data, total_transform_matrix)

        fig = plt.figure(figsize=(24, 12))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1920)
        ax.set_ylim(800, 0)
        ax.set_aspect("equal")

        # 将移动前的gaze_point用橙色标记。
        ax.scatter(gaze_point_list_1d[:, 0], gaze_point_list_1d[:, 1], c='orange', marker='o', s=1, zorder=1)

        # update gaze_point_list_1d
        gaze_point_list_1d = [UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_point) for gaze_point in gaze_point_list_1d]
        gaze_point_list_1d = [np.dot(transform_matrix, gaze_point) for gaze_point in gaze_point_list_1d]
        gaze_point_list_1d = np.array([UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_point_list_1d])
        # update reading_data
        for text_index in range(len(reading_data)):
            gaze_coordinates = reading_data[text_index][["gaze_x", "gaze_y"]].values.tolist()
            gaze_coordinates = [UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates = [np.dot(transform_matrix, gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates = [UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            reading_data[text_index][["gaze_x", "gaze_y"]] = gaze_coordinates

        # 将移动后的gaze_point用绿色标记。
        ax.scatter(gaze_point_list_1d[:, 0], gaze_point_list_1d[:, 1], c='g', marker='o', s=1, zorder=1)

        # TODO 这里先简单写一个看效果的demo，之后再将函数做合适的封装处理。
        distance_list, avg_distance = ManualCalibrateForStd.compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        avg_error_list.append(avg_distance)
        print(f"average distance: {avg_distance}, last iteration num: {last_iteration_num}")
        last_iteration_num_list.append(last_iteration_num)

        max_pair_num = max(actual_text_point_dict.values())
        for key, value in actual_text_point_dict.items():
            color_ratio = 0.8 - (value / max_pair_num) * 0.6
            color = (color_ratio, color_ratio, color_ratio)
            if value == 0:
                ax.scatter(key[0], key[1], c=color, marker='x', s=40, zorder=2)
            else:
                ax.scatter(key[0], key[1], c=color, marker='o', s=40, zorder=2)

        max_pair_num = max(actual_supplement_text_point_dict.values())
        for key, value in actual_supplement_text_point_dict.items():
            color_ratio = 0.8 - (value / max_pair_num) * 0.6
            color = (1, color_ratio, color_ratio)
            if value == 0:
                ax.scatter(key[0], key[1], c=color, marker='x', s=10, zorder=3)
            else:
                ax.scatter(key[0], key[1], c=color, marker='o', s=10, zorder=3)

        color_list = []
        line_segment_list = []
        for point_pair_index in range(len(point_pair_list)):
            # point_pair_gaze = UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair_list[point_pair_index][0])
            # point_pair_gaze = np.dot(transform_matrix, point_pair_gaze)
            # point_pair_gaze = UtilFunctions.change_homogeneous_vector_to_2d_vector(point_pair_gaze)
            if weight_list[point_pair_index] > 0:
                color_list.append("b")
            else:
                color_list.append("r")
            # point_pair_gaze = last_point_pair_list[point_pair_index][0]
            line_segment_list.append([point_pair_list[point_pair_index][0], point_pair_list[point_pair_index][1]])
        line_collection = LineCollection(line_segment_list, colors=color_list, linewidths=0.5, zorder=0)
        ax.add_collection(line_collection)

        # plt.show()
        # plt.clf()
        # plt.close()
        # Render.visualize_cali_result(gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list,
        #                              avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list,
        #                              calibration_point_list_modified, file_name=None)

        gaze_file_path = f"pic/reading_matching/gaze_matching/subject_{subject_index}"
        if not os.path.exists(gaze_file_path):
            os.makedirs(gaze_file_path)
        plt.savefig(f"{gaze_file_path}/iteration_{iteration_index}.png")
        plt.clf()
        plt.close()
        calibration_file_path = f"pic/reading_matching/calibration/subject_{subject_index}"
        if not os.path.exists(calibration_file_path):
            os.makedirs(calibration_file_path)
        Render.visualize_cali_result(gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list,
                                     avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list,
                                     calibration_point_list_modified, file_name=f"{calibration_file_path}/iteration_{iteration_index}.png")

    log_path = "log/gradient_descent_avg_error"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = open(f"{log_path}/subject_{subject_index}.txt", "w")
    log_file.write(f"with_batch: False, "
                   f"location_penalty: {configs.location_penalty}, punctuation_penalty: {configs.punctuation_penalty}, "
                   f"empty_penalty: {configs.empty_penalty}, completion_weight: {configs.completion_weight},"
                   f"right_down_corner_unmatched_ratio: {configs.right_down_corner_unmatched_ratio}, "
                   f"left_boundary_ratio: {configs.left_boundary_ratio}, right_boundary_ratio: {configs.right_boundary_ratio}, "
                   f"left_boundary_distance_threshold_ratio: {configs.left_boundary_distance_threshold_ratio}, "
                   f"right_boundary_distance_threshold_ratio: {configs.right_boundary_distance_threshold_ratio}, "
                   f"gradient_descent_stop_accuracy: {configs.gradient_descent_stop_accuracy}\n")

    for iteration_index in range(len(avg_error_list)):
        print("avg_error_list[", iteration_index, "]: ", avg_error_list[iteration_index])
        log_file.write(
            f"avg_error_list[{iteration_index}]: {avg_error_list[iteration_index]}, last_iteration_num: {last_iteration_num_list[iteration_index]}, "
            f"last_gd_error: {gd_error_list[iteration_index]}, learning_rate: {learning_rate_list[iteration_index]}\n")

    log_file.close()

    return avg_error_list


def calibrate_with_simple_linear_weight(subject_index, reading_data, text_data, calibration_data, max_iteration=100, distance_threshold=64):
    reading_data = reading_data.copy()

    # 获取1d的gaze point list。
    total_gaze_point_num = 0
    gaze_point_list_1d = []
    for text_index in range(len(reading_data)):
        coordinates = reading_data[text_index][["gaze_x", "gaze_y"]].values.tolist()
        gaze_point_list_1d.extend(coordinates)
        total_gaze_point_num += len(coordinates)
    gaze_point_list_1d = np.array(gaze_point_list_1d)

    filtered_text_data_list = [[] for _ in range(len(text_data))]
    for text_index in range(len(text_data)):
        for row_index in range(configs.row_num):
            df = text_data[text_index]
            filtered_text_data_df = df[(df["row"] == float(row_index)) & (df["word"] != "blank_supplement")]
            filtered_text_data_list[text_index].append(filtered_text_data_df)

    # 生成3个dict。text_point_dict用来记录所有有效的文本点；effective_text_point_dict用来记录非blank_supplement，且至少有过一次文字的文本点。supplement_text_point_dict用来记录blank_supplement的文本点。
    text_point_dict, effective_text_point_dict, supplement_text_point_dict, text_point_total_utilized_count = prepare_dicts_for_text_point(subject_index, text_data, calibration_data)
    # 生成：按文本、行号来对text point分类得到的nearest neighbor；按文本、行号来对text point分类，且去除blank_supplement的nearest neighbor；以及所有text_point的nearest neighbor。
    row_nbrs_list, total_nbrs_list = create_text_nearest_neighbor(text_data)
    # 将gaze data和text point先做缩放，然后基于重心对齐。
    gaze_point_list_1d, effective_text_point_dict, reading_data, calibration_data = transform_using_centroid_and_outbound(gaze_point_list_1d, effective_text_point_dict, reading_data, subject_index, calibration_data)

    total_transform_matrix = np.eye(3)
    avg_error_list = []
    last_iteration_num_list = []
    last_iteration_num = 100000
    gd_error_list = []
    learning_rate_list = []
    last_grad_norm = 10000
    for iteration_index in range(max_iteration):
        print("iteration_index: ", iteration_index)
        # 每次迭代前，创建一个类似effective_text_point_dict的字典，用于记录每个文本点被阅读点覆盖的次数。
        actual_text_point_dict = effective_text_point_dict.copy()
        for key in actual_text_point_dict:
            actual_text_point_dict[key] = 0

        actual_supplement_text_point_dict = supplement_text_point_dict.copy()
        point_pair_list, weight_list, row_label_list = point_matching_3(reading_data, gaze_point_list_1d, text_data, filtered_text_data_list,
                                                                        total_nbrs_list, row_nbrs_list,
                                                                        effective_text_point_dict, actual_text_point_dict, actual_supplement_text_point_dict,
                                                                        distance_threshold)

        # print(f"first 10 point pairs: {point_pair_list[:10]}\n"
        #       f"last 10 point pairs: {point_pair_list[-10:]}\n")

        # fig = plt.figure(figsize=(24, 12))
        # ax = fig.add_subplot(111)
        # ax.set_xlim(0, 1920)
        # ax.set_ylim(800, 0)
        # ax.set_aspect("equal")
        # color_list = [(0.5, 0.5, 0), (0, 1, 0), (0.5, 0.5, 1), (1, 0.5, 0.5), (0.5, 1, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0.5)]
        # ax.scatter(np.array(point_pair_list)[:, 0, 0], np.array(point_pair_list)[:, 0, 1], c=[color_list[i] for i in row_label_list], marker='o', s=1, zorder=1)
        # ax.scatter(np.array(point_pair_list)[:, 1, 0], np.array(point_pair_list)[:, 1, 1], c=[color_list[i] for i in row_label_list], marker='x', s=10, zorder=1)
        # line_segment_list = []
        # color_list = []
        # for point_pair_index in range(len(point_pair_list)):
        #     line_segment_list.append([point_pair_list[point_pair_index][0], point_pair_list[point_pair_index][1]])
        #     if weight_list[point_pair_index] > 0:
        #         color_list.append("b")
        #     else:
        #         color_list.append("r")
        # line_collection = LineCollection(line_segment_list, colors=color_list, linewidths=0.5, zorder=0)
        # ax.add_collection(line_collection)
        # plt.show()

        learning_rate = 1

        # if iteration_index > int(max_iteration / 2):
        #     learning_rate = 1e-2
        learning_rate_list.append(learning_rate)

        # transform_matrix, gd_error, last_iteration_num = GradientDescent.gradient_descent_with_torch(point_pair_list, weight_list, learning_rate=learning_rate, last_iteration_num=last_iteration_num, max_iterations=5000, grad_clip_value=0, stop_grad_norm=5)
        transform_matrix, gd_error, last_iteration_num, last_grad_norm \
            = GradientDescent.gradient_descent_with_torch(point_pair_list, weight_list,
                                                          learning_rate=learning_rate, last_iteration_num=last_iteration_num,
                                                          max_iterations=5000, stop_grad_norm=1, grad_clip_value=1e8)

        gd_error_list.append(gd_error)
        # print(f"transform_matrix: {transform_matrix}")
        # update total_transform_matrix
        total_transform_matrix = np.dot(transform_matrix, total_transform_matrix)

        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
            avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
            calibration_point_list_modified = ManualCalibrateForStd.apply_transform_to_calibration(subject_index, calibration_data, total_transform_matrix)

        fig = plt.figure(figsize=(24, 12))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1920)
        ax.set_ylim(800, 0)
        ax.set_aspect("equal")

        # 将移动前的gaze_point用橙色标记。
        ax.scatter(gaze_point_list_1d[:, 0], gaze_point_list_1d[:, 1], c='orange', marker='o', s=1, zorder=1)

        # update gaze_point_list_1d
        gaze_point_list_1d = [UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_point) for gaze_point in gaze_point_list_1d]
        gaze_point_list_1d = [np.dot(transform_matrix, gaze_point) for gaze_point in gaze_point_list_1d]
        gaze_point_list_1d = np.array([UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_point_list_1d])
        # update reading_data
        for text_index in range(len(reading_data)):
            gaze_coordinates = reading_data[text_index][["gaze_x", "gaze_y"]].values.tolist()
            gaze_coordinates = [UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates = [np.dot(transform_matrix, gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates = [UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            reading_data[text_index][["gaze_x", "gaze_y"]] = gaze_coordinates

        # 将移动后的gaze_point用绿色标记。
        ax.scatter(gaze_point_list_1d[:, 0], gaze_point_list_1d[:, 1], c='g', marker='o', s=1, zorder=1)

        # TODO 这里先简单写一个看效果的demo，之后再将函数做合适的封装处理。
        distance_list, avg_distance = ManualCalibrateForStd.compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        avg_error_list.append(avg_distance)
        print(f"average distance: {avg_distance}, last iteration num: {last_iteration_num}")
        last_iteration_num_list.append(last_iteration_num)

        max_pair_num = max(actual_text_point_dict.values())
        for key, value in actual_text_point_dict.items():
            color_ratio = 0.8 - (value / max_pair_num) * 0.6
            color = (color_ratio, color_ratio, color_ratio)
            if value == 0:
                ax.scatter(key[0], key[1], c=color, marker='x', s=40, zorder=2)
            else:
                ax.scatter(key[0], key[1], c=color, marker='o', s=40, zorder=2)

        # max_pair_num = max(actual_supplement_text_point_dict.values())
        # for key, value in actual_supplement_text_point_dict.items():
        #     color_ratio = 0.8 - (value / max_pair_num) * 0.6
        #     color = (1, color_ratio, color_ratio)
        #     if value == 0:
        #         ax.scatter(key[0], key[1], c=color, marker='x', s=10, zorder=3)
        #     else:
        #         ax.scatter(key[0], key[1], c=color, marker='o', s=10, zorder=3)

        color_list = []
        line_segment_list = []
        weight_above_0 = weight_list.copy()
        weight_above_0 = [weight for weight in weight_above_0 if weight > 0]
        weight_max_above_0 = np.percentile(weight_above_0, 90)
        weight_min_above_0 = np.percentile(weight_above_0, 10)
        weight_difference_above_0 = weight_max_above_0 - weight_min_above_0
        weight_below_0 = weight_list.copy()
        weight_below_0 = [-weight for weight in weight_below_0 if weight < 0]
        if len(weight_below_0) > 0:
            weight_max_below_0 = np.percentile(weight_below_0, 90)
            weight_min_below_0 = np.percentile(weight_below_0, 10)
            weight_difference_below_0 = weight_max_below_0 - weight_min_below_0
        for point_pair_index in range(len(point_pair_list)):
            if weight_list[point_pair_index] > 0:
                color_ratio = 0.5 - min(((weight_list[point_pair_index] - weight_min_above_0) / weight_difference_above_0) * 0.4, 0.4)
                color_list.append((color_ratio, color_ratio, color_ratio + 0.5))
            else:
                if len(weight_below_0) > 0:
                    color_ratio = 0.5 - min(((-weight_list[point_pair_index] - weight_min_below_0) / weight_difference_below_0) * 0.4, 0.4)
                    color_list.append((color_ratio + 0.5, color_ratio, color_ratio))
                else:
                    color_list.append((1, 0, 0))
            line_segment_list.append([point_pair_list[point_pair_index][0], point_pair_list[point_pair_index][1]])
        line_collection = LineCollection(line_segment_list, colors=color_list, linewidths=1, zorder=3)
        ax.add_collection(line_collection)

        # plt.show()
        # plt.clf()
        # plt.close()
        # Render.visualize_cali_result(gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list,
        #                              avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list,
        #                              calibration_point_list_modified, file_name=None)

        gaze_file_path = f"pic/reading_matching/{configs.file_index}/gaze_matching/subject_{subject_index}"
        if not os.path.exists(gaze_file_path):
            os.makedirs(gaze_file_path)
        plt.savefig(f"{gaze_file_path}/iteration_{iteration_index}.png")
        plt.clf()
        plt.close()
        calibration_file_path = f"pic/reading_matching/{configs.file_index}/calibration/subject_{subject_index}"
        if not os.path.exists(calibration_file_path):
            os.makedirs(calibration_file_path)
        Render.visualize_cali_result(gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list,
                                     avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list,
                                     calibration_point_list_modified, file_name=f"{calibration_file_path}/iteration_{iteration_index}.png")

    log_path = f"log/{configs.file_index}/gradient_descent_avg_error"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = open(f"{log_path}/subject_{subject_index}.txt", "w")
    log_file.write(f"with_batch: False, "
                   f"location_penalty: {configs.location_penalty}, punctuation_penalty: {configs.punctuation_penalty}, "
                   f"empty_penalty: {configs.empty_penalty}, completion_weight: {configs.completion_weight},"
                   f"right_down_corner_unmatched_ratio: {configs.right_down_corner_unmatched_ratio}, "
                   f"left_boundary_ratio: {configs.left_boundary_ratio}, right_boundary_ratio: {configs.right_boundary_ratio}, "
                   f"left_boundary_distance_threshold_ratio: {configs.left_boundary_distance_threshold_ratio}, "
                   f"right_boundary_distance_threshold_ratio: {configs.right_boundary_distance_threshold_ratio}, "
                   f"gradient_descent_stop_accuracy: {configs.gradient_descent_stop_accuracy},"
                   f"weight_divisor: {configs.weight_divisor}, weight_intercept: {configs.weight_intercept}\n")

    for iteration_index in range(len(avg_error_list)):
        print("avg_error_list[", iteration_index, "]: ", avg_error_list[iteration_index])
        log_file.write(
            f"avg_error_list[{iteration_index}]: {avg_error_list[iteration_index]}, last_iteration_num: {last_iteration_num_list[iteration_index]}, "
            f"last_gd_error: {gd_error_list[iteration_index]}, learning_rate: {learning_rate_list[iteration_index]}\n")

    log_file.close()

    return avg_error_list


def calibrate_with_simple_linear_weight_ls(subject_index, reading_data, text_data, calibration_data, max_iteration=100, distance_threshold=64):
    reading_data = reading_data.copy()

    # 获取1d的gaze point list。
    total_gaze_point_num = 0
    gaze_point_list_1d = []
    for text_index in range(len(reading_data)):
        coordinates = reading_data[text_index][["gaze_x", "gaze_y"]].values.tolist()
        gaze_point_list_1d.extend(coordinates)
        total_gaze_point_num += len(coordinates)
    gaze_point_list_1d = np.array(gaze_point_list_1d)

    filtered_text_data_list = [[] for _ in range(len(text_data))]
    for text_index in range(len(text_data)):
        for row_index in range(configs.row_num):
            df = text_data[text_index]
            filtered_text_data_df = df[(df["row"] == float(row_index)) & (df["word"] != "blank_supplement")]
            filtered_text_data_list[text_index].append(filtered_text_data_df)

    # 生成3个dict。text_point_dict用来记录所有有效的文本点；effective_text_point_dict用来记录非blank_supplement，且至少有过一次文字的文本点。supplement_text_point_dict用来记录blank_supplement的文本点。
    text_point_dict, effective_text_point_dict, supplement_text_point_dict, text_point_total_utilized_count = prepare_dicts_for_text_point(subject_index, text_data, calibration_data)
    # 生成：按文本、行号来对text point分类得到的nearest neighbor；按文本、行号来对text point分类，且去除blank_supplement的nearest neighbor；以及所有text_point的nearest neighbor。
    row_nbrs_list, total_nbrs_list = create_text_nearest_neighbor(text_data)
    # 将gaze data和text point先做缩放，然后基于重心对齐。
    gaze_point_list_1d, effective_text_point_dict, reading_data, calibration_data = transform_using_centroid_and_outbound(gaze_point_list_1d, effective_text_point_dict, reading_data, subject_index,
                                                                                                                          calibration_data)

    total_transform_matrix = np.eye(3)
    avg_error_list = []
    last_iteration_num_list = []
    last_iteration_num = 100000
    gd_error_list = []
    learning_rate_list = []
    last_grad_norm = 10000
    for iteration_index in range(max_iteration):
        print("iteration_index: ", iteration_index)
        # 每次迭代前，创建一个类似effective_text_point_dict的字典，用于记录每个文本点被阅读点覆盖的次数。
        actual_text_point_dict = effective_text_point_dict.copy()
        for key in actual_text_point_dict:
            actual_text_point_dict[key] = 0

        actual_supplement_text_point_dict = supplement_text_point_dict.copy()
        point_pair_list, weight_list, row_label_list = point_matching_3(reading_data, gaze_point_list_1d, text_data, filtered_text_data_list,
                                                                        total_nbrs_list, row_nbrs_list,
                                                                        effective_text_point_dict, actual_text_point_dict, actual_supplement_text_point_dict,
                                                                        distance_threshold)
        point_pair_list = np.array(point_pair_list)
        source_points = point_pair_list[:, 0, :]
        target_points = point_pair_list[:, 1, :]
        source_points = source_points.astype(np.float32)
        target_points = target_points.astype(np.float32)
        affine_matrix, _ = cv2.estimateAffine2D(source_points, target_points)
        affine_matrix = np.vstack((affine_matrix, np.array([0, 0, 1])))
        source_points_extend = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
        source_point_extend_after_transform = np.dot(affine_matrix, source_points_extend.T).T
        source_point_after_transform = source_point_extend_after_transform[:, :2]
        error = np.linalg.norm(source_point_after_transform - target_points, axis=1)
        avg_error = np.mean(error)
        print(f"iteration: {iteration_index}, error: {avg_error}")

        # update gaze_point_list_1d
        gaze_point_list_1d = [UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_point) for gaze_point in gaze_point_list_1d]
        gaze_point_list_1d = [np.dot(affine_matrix, gaze_point) for gaze_point in gaze_point_list_1d]
        gaze_point_list_1d = np.array([UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_point) for gaze_point in gaze_point_list_1d])
        # update reading_data
        for text_index in range(len(reading_data)):
            gaze_coordinates = reading_data[text_index][["gaze_x", "gaze_y"]].values.tolist()
            gaze_coordinates = [UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates = [np.dot(affine_matrix, gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates = [UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            reading_data[text_index][["gaze_x", "gaze_y"]] = gaze_coordinates

    return avg_error_list


def calibrate_reading_with_whole_matrix_gradient_descent(subject_index, reading_data, text_data, calibration_data, max_iteration=100, distance_threshold=64, mode="location"):
    avg_error_list = []
    if mode == "location":
        avg_error_list = calibrate_with_location(subject_index, reading_data, text_data, calibration_data, max_iteration=max_iteration, distance_threshold=distance_threshold)
    elif mode == "location_and_coverage":
        avg_error_list = calibrate_with_location_and_coverage(subject_index, reading_data, text_data, calibration_data, max_iteration=max_iteration, distance_threshold=distance_threshold)
    elif mode == "location_coverage_and_penalty":
        avg_error_list = calibrate_with_location_coverage_and_penalty(subject_index, reading_data, text_data, calibration_data, max_iteration=max_iteration, distance_threshold=distance_threshold)
    elif mode == "location_coverage_penalty_and_rowlabel":
        avg_error_list = calibrate_with_location_coverage_penalty_and_rowlabel(subject_index, reading_data, text_data, calibration_data, max_iteration=max_iteration, distance_threshold=distance_threshold)
    elif mode == "torch":
        avg_error_list = calibrate_with_torch(subject_index, reading_data, text_data, calibration_data, max_iteration=max_iteration, distance_threshold=distance_threshold)
    elif mode == "simple_linear_weight":
        avg_error_list = calibrate_with_simple_linear_weight(subject_index, reading_data, text_data, calibration_data, max_iteration=max_iteration, distance_threshold=distance_threshold)
    elif mode == "simple_linear_weight_ls":
        avg_error_list = calibrate_with_simple_linear_weight_ls(subject_index, reading_data, text_data, calibration_data, max_iteration=max_iteration, distance_threshold=distance_threshold)

    return avg_error_list


