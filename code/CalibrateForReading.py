import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from sklearn.neighbors import NearestNeighbors
import GradientDescent
import ManualCalibrateForStd
import Render
import UtilFunctions
import configs


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

    # 将gaze data和text point先做一个基于重心的对齐。
    gaze_point_center = np.mean(gaze_point_list_1d, axis=0)
    text_point_center = np.array([0, 0])
    for key, value in effective_text_point_dict.items():
        text_point_center[0] += key[0]
        text_point_center[1] += key[1]
    text_point_center[0] /= len(effective_text_point_dict)
    text_point_center[1] /= len(effective_text_point_dict)
    translate_vector = np.array(text_point_center - gaze_point_center)
    gaze_point_list_1d += translate_vector
    for text_index in range(len(reading_data)):
        reading_data[text_index]["gaze_x"] += translate_vector[0]
        reading_data[text_index]["gaze_y"] += translate_vector[1]

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
    with multiprocessing.Pool(processes=configs.number_of_process) as pool:
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

            t1 = time.time()
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
            t2 = time.time()
            print("time: ", t2 - t1)

            fig = plt.figure(figsize=(24, 12))
            ax = fig.add_subplot(111)
            ax.set_xlim(0, 1920)
            ax.set_ylim(800, 0)
            ax.set_aspect("equal")
            color_list = [(1, 0, 0), (0, 1, 0), (0.5, 0.5, 1), (1, 0.5, 0.5), (0.5, 1, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0.5)]
            ax.scatter(np.array(point_pair_list)[:, 0, 0], np.array(point_pair_list)[:, 0, 1], c=[color_list[i] for i in row_label_list], marker='o', s=1, zorder=1)
            ax.scatter(np.array(point_pair_list)[:, 1, 0], np.array(point_pair_list)[:, 1, 1], c=[color_list[i] for i in row_label_list], marker='x', s=10, zorder=1)
            line_segment_list = []
            for point_pair_index in range(len(point_pair_list)):
                line_segment_list.append([point_pair_list[point_pair_index][0], point_pair_list[point_pair_index][1]])
            line_collection = LineCollection(line_segment_list, colors='b', linewidths=0.5, zorder=0)
            ax.add_collection(line_collection)
            plt.show()

            transform_matrix, gd_error = GradientDescent.gradient_descent_with_whole_matrix_using_tensor_with_weight(point_pair_list, weight_list, max_iterations=1000)

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

            # update total_transform_matrix
            total_transform_matrix = np.dot(transform_matrix, total_transform_matrix)

            # 将移动后的gaze_point用绿色标记。
            ax.scatter(gaze_point_list_1d[:, 0], gaze_point_list_1d[:, 1], c='g', marker='o', s=1, zorder=1)

            # TODO 这里先简单写一个看效果的demo，之后再将函数做合适的封装处理。
            distance_list, avg_distance = ManualCalibrateForStd.compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
            avg_error_list.append(avg_distance)
            print("average distance: ", avg_distance)

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

    # log_file = open(f"log/gradient_descent_avg_error/subject_{subject_index}.txt", "w")

    # for iteration_index in range(len(avg_error_list)):
    #     print("avg_error_list[", iteration_index, "]: ", avg_error_list[iteration_index])
    #     log_file.write(f"avg_error_list[{iteration_index}]: {avg_error_list[iteration_index]}\n")
    #
    # log_file.close()

    return avg_error_list


def calibrate_reading_with_whole_matrix_gradient_descent(subject_index, reading_data, text_data, calibration_data, max_iteration=100, distance_threshold=32, mode="location"):
    avg_error_list = []
    if mode == "location":
        avg_error_list = calibrate_with_location(subject_index, reading_data, text_data, calibration_data, max_iteration=max_iteration, distance_threshold=distance_threshold)
    elif mode == "location_and_coverage":
        avg_error_list = calibrate_with_location_and_coverage(subject_index, reading_data, text_data, calibration_data, max_iteration=max_iteration, distance_threshold=distance_threshold)
    elif mode == "location_coverage_and_penalty":
        avg_error_list = calibrate_with_location_coverage_and_penalty(subject_index, reading_data, text_data, calibration_data, max_iteration=max_iteration, distance_threshold=distance_threshold)
    elif mode == "location_coverage_penalty_and_rowlabel":
        avg_error_list = calibrate_with_location_coverage_penalty_and_rowlabel(subject_index, reading_data, text_data, calibration_data, max_iteration=max_iteration, distance_threshold=distance_threshold)

    return avg_error_list


