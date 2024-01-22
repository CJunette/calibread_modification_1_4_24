import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree

import ComputeTextDensity
import Render
import SaveFiles
import UtilFunctions
import GradientDescent
import ManualCalibrateForStd
import ReadData
import configs


def change_2d_vector_to_homogeneous_vector(vector_2d):
    vector_homogeneous = np.array([vector_2d[0], vector_2d[1], 1])
    return vector_homogeneous


def change_homogeneous_vector_to_2d_vector(vector_homogeneous):
    # vector_2d = [vector_homogeneous[i] / vector_homogeneous[-1] for i in range(len(vector_homogeneous))]
    # vector_2d = np.array(vector_2d[:-1])
    vector_2d = np.array(vector_homogeneous[:-1])
    return vector_2d


def get_paired_points_of_std_cali_from_cali_dict(avg_gaze_list, calibration_point_list):
    '''
    从ReadData.read_calibration_data()得到的数据中，对avg_gaze与calibration_point进行配对。
    :param avg_gaze_list:
    :param calibration_point_list:
    :return:
    '''
    point_pairs = []

    for row_index in range(len(avg_gaze_list)):
        for col_index in range(len(avg_gaze_list[row_index])):
            avg_calibration_point_dict = avg_gaze_list[row_index][col_index]
            calibration_point_dict = calibration_point_list[row_index][col_index]
            avg_point = [avg_calibration_point_dict["avg_gaze_x"], avg_calibration_point_dict["avg_gaze_y"]]
            calibration_point = [calibration_point_dict["point_x"], calibration_point_dict["point_y"]]
            point_pairs.append([avg_point, calibration_point])

    return np.array(point_pairs)


def compare_manual_calibration_errors(method_1, method_2):
    '''
    比较icp和gradient descent的标定误差。
    :return:
    '''

    calibration_data = ReadData.read_calibration_data()

    # visualize the manual calibration data after icp calibration.
    method_1_avg_distance_list = []
    for subject_index in range(0, 19):
        method_1_transform_matrix = method_1(subject_index, calibration_data)

        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
        avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
        calibration_point_list_modified = ManualCalibrateForStd.apply_transform_to_calibration(subject_index, calibration_data, method_1_transform_matrix)

        distance_list, avg_distance = ManualCalibrateForStd.compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        print("avg_distance: ", avg_distance)
        method_1_avg_distance_list.append(avg_distance)
        # Render.visualize_cali_result(gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list,
        #                              avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list,
        #                              calibration_point_list_modified)

    # visualize the manual calibration data after gradient descent calibration.
    method_2_avg_distance_list = []
    for subject_index in range(0, 19):
        method_2_transform_matrix = method_2(subject_index, calibration_data)
        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
        avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
        calibration_point_list_modified = ManualCalibrateForStd.apply_transform_to_calibration(subject_index, calibration_data, method_2_transform_matrix)

        distance_list, avg_distance = ManualCalibrateForStd.compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        print("avg_distance: ", avg_distance)
        method_2_avg_distance_list.append(avg_distance)
        # Render.visualize_cali_result(gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list,
        #                              avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list,
        #                              calibration_point_list_modified)

    for i in range(len(method_1_avg_distance_list)):
        print(f"{method_1_avg_distance_list[i]:.2f}", end=", ")
    print()
    for i in range(len(method_2_avg_distance_list)):
        print(f"{method_2_avg_distance_list[i]:.2f}", end=", ")


def visualize_manual_calibration(method):
    '''
    visualize the manual calibration data after icp calibration.
    :return:
    '''
    calibration_data = ReadData.read_calibration_data()
    avg_distance_list = []
    for subject_index in range(0, 19):
        transform_matrix = method(subject_index, calibration_data)

        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
        avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
        calibration_point_list_modified = ManualCalibrateForStd.apply_transform_to_calibration(subject_index, calibration_data, transform_matrix)

        distance_list, avg_distance = ManualCalibrateForStd.compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        print("avg_distance: ", avg_distance)
        avg_distance_list.append(avg_distance)
        Render.visualize_cali_result(gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list,
                                     avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list,
                                     calibration_point_list_modified)

    return avg_distance_list


def trim_data(read_gaze_data, calibration_data):
    '''

    :param read_gaze_data:
    :param calibration_data:
    :return:
      read_gaze_data_after_transform_list_1: 移动之后的坐标。
      read_gaze_data_after_trim_list_1：移动，裁剪之后的坐标。
      read_gaze_data_after_restore_list_：移动，裁剪，复位之后的坐标。
    '''
    transform_matrix_list = []
    read_gaze_data_after_transform_list_1 = []
    read_gaze_data_after_trim_list_1 = []
    read_gaze_data_after_restore_list_1 = []

    for subject_index in range(len(read_gaze_data)):
        transform_matrix = ManualCalibrateForStd.compute_std_cali_with_homography_matrix(subject_index, calibration_data)
        transform_matrix_list.append(transform_matrix)

        read_gaze_data_after_transform_list_2 = []
        read_gaze_data_after_trim_list_2 = []
        read_gaze_data_after_restore_list_2 = []

        for reading_index in range(len(read_gaze_data[subject_index])):
            df = read_gaze_data[subject_index][reading_index]
            gaze_x = df["gaze_x"].tolist()
            gaze_y = df["gaze_y"].tolist()
            gaze_x = [float(x) for x in gaze_x]
            gaze_y = [float(y) for y in gaze_y]
            gaze_coordinates = [[gaze_x[i], gaze_y[i]] for i in range(len(gaze_x))]
            gaze_coordinates = np.array(gaze_coordinates)

            # transform data using homography matrix.
            df_after_transform = df.copy()
            gaze_coordinates_after_transform = [np.dot(transform_matrix, UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_coordinate)) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates_after_transform = [UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates_after_transform]
            df_after_transform["gaze_x"] = np.array(gaze_coordinates_after_transform)[:, 0]
            df_after_transform["gaze_y"] = np.array(gaze_coordinates_after_transform)[:, 1]
            read_gaze_data_after_transform_list_2.append(df_after_transform)

            # iterate over gaze_coordinates_after_transform, if the coordinate x, y satisfy x < 340 or x > 1580 or y < 176 or y > 624. select them out in df_after_trim (pd.DataFrame).
            keep_index_after_trim_list = []
            gaze_coordinates_after_trim = []
            for gaze_coordinate_index in range(len(gaze_coordinates_after_transform)):
                gaze_coordinate = gaze_coordinates_after_transform[gaze_coordinate_index]
                if 340 <= gaze_coordinate[0] <= 1580 and 208 <= gaze_coordinate[1] <= 624:
                    keep_index_after_trim_list.append(gaze_coordinate_index)
                    gaze_coordinates_after_trim.append(gaze_coordinate)
            df_after_trim = df_after_transform.iloc[keep_index_after_trim_list]
            df_after_trim.reset_index(drop=True, inplace=True)
            read_gaze_data_after_trim_list_2.append(df_after_trim)

            # move data back using the inverse of homography matrix.
            gaze_coordinates_after_restore = [np.dot(np.linalg.inv(transform_matrix), UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_coordinate)) for gaze_coordinate in gaze_coordinates_after_trim]
            gaze_coordinates_after_restore = [UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates_after_restore]
            df_after_restore = df_after_trim.copy()
            df_after_restore["gaze_x"] = np.array(gaze_coordinates_after_restore)[:, 0]
            df_after_restore["gaze_y"] = np.array(gaze_coordinates_after_restore)[:, 1]
            df_after_restore.rename(columns={"Unnamed: 0": "original_index"}, inplace=True)
            read_gaze_data_after_restore_list_2.append(df_after_restore)

        read_gaze_data_after_transform_list_1.append(read_gaze_data_after_transform_list_2)
        read_gaze_data_after_trim_list_1.append(read_gaze_data_after_trim_list_2)
        read_gaze_data_after_restore_list_1.append(read_gaze_data_after_restore_list_2)

    return read_gaze_data_after_transform_list_1, read_gaze_data_after_trim_list_1, read_gaze_data_after_restore_list_1


def compute_density_for_reading_data(reading_data, distance_threshold=40):
    for subject_index in range(len(reading_data)):
        for text_index in range(len(reading_data[subject_index])):
            df = reading_data[subject_index][text_index]
            gaze_x = df["gaze_x"].tolist()
            gaze_y = df["gaze_y"].tolist()
            gaze_x = [float(x) for x in gaze_x]
            gaze_y = [float(y) for y in gaze_y]
            gaze_coordinates = [[gaze_x[i], gaze_y[i]] for i in range(len(gaze_x))]
            gaze_coordinates = np.array(gaze_coordinates)

            gaze_tree = cKDTree(gaze_coordinates)
            neighbors_count = [len(gaze_tree.query_ball_point(point, distance_threshold)) - 1 for point in gaze_coordinates]
            reading_data[subject_index][text_index]["density"] = neighbors_count

    return reading_data


def supplement_bound_points(para_id, start_col, end_col, start_row, offset_x=-1, offset_y=-0.5):
    new_point_list = []
    row = start_row + offset_y
    start_x = configs.left_top_text_center[0]
    start_y = configs.left_top_text_center[1] + start_row * configs.text_height
    y = start_y + offset_y * configs.text_height

    for index in range(start_col, end_col):
        x = start_x + offset_x * configs.text_width + index * configs.text_width
        col = offset_x + index
        new_point = {"para_id": para_id,
                     "x": x, "y": y,
                     "row": row, "col": col,
                     "word": "blank_supplement"}
        new_point_list.append(new_point)

    return new_point_list


def add_boundary_points_to_text_data(text_data):
    '''
    在每个文字点的周围，添加一些边界点。这些边界点的坐标为：*.5。
    :param text_data:
    :return:
    '''

    for text_index in range(len(text_data)):
        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(111)
        # ax.set_xlim(0, 1920)
        # ax.set_ylim(1080, 0)
        # ax.set_aspect('equal', adjustable='box')

        new_point_list = []
        df = text_data[text_index]

        # for index, row in df.iterrows():
        #     x = row["x"]
        #     y = row["y"]
        #     ax.scatter(x, y, c="black", marker='o', s=10, zorder=10)

        row_list = df["row"].unique().tolist()
        row_list.sort()
        para_id = df["para_id"].tolist()[0]
        for row_index in range(len(row_list)):
            col_list = df[df["row"] == row_list[row_index]]["col"].unique().tolist()
            col_list.sort()

            new_points = []

            left_right_padding = 3
            up_down_padding = 1

            # 如果某一行内本身就存在一些空白点，则将其添加进去。如果没有空白点，则添加首尾的边界点。
            min_col = min(col_list)
            points = supplement_bound_points(para_id=para_id, start_col=-left_right_padding, end_col=min_col, start_row=row_list[row_index], offset_x=0, offset_y=0)
            new_points.extend(points)
            max_col = max(col_list)
            points = supplement_bound_points(para_id=para_id, start_col=max_col + 1, end_col=configs.col_num+left_right_padding, start_row=row_list[row_index], offset_x=0, offset_y=0)
            new_points.extend(points)

            # 除此之外，额外添加行与行之间、以及文字边缘的点。目前的设计是每一行添加其上方的点。
            # if row_index > 0:
            #     points = supplement_bound_points(para_id=para_id, start_col=-left_right_padding, end_col=configs.col_num + left_right_padding, start_row=row_list[row_index], offset_x=0, offset_y=-0.5)
            #     new_points.extend(points)

            if row_index == 0:
                for repeat_index in range(1, up_down_padding+1):
                    points = supplement_bound_points(para_id=para_id, start_col=-left_right_padding, end_col=configs.col_num + left_right_padding, start_row=row_list[row_index], offset_x=0, offset_y=-0.5*repeat_index)
                    new_points.extend(points)

            # 如果是最后一行，则还需要添加下方的点。
            if row_index == len(row_list) - 1:
                for repeat_index in range(1, up_down_padding+1):
                    points = supplement_bound_points(para_id=para_id, start_col=-left_right_padding, end_col=configs.col_num + left_right_padding, start_row=row_list[row_index], offset_x=0, offset_y=0.5*repeat_index)
                    new_points.extend(points)

            new_point_list.extend(new_points)

        # add new_point_list to df.
        df = df.append(new_point_list, ignore_index=True)

        # for index, row in df.iterrows():
        #     x = row["x"]
        #     y = row["y"]
        #     ax.scatter(x, y, c="red", marker='o', s=5)
        # plt.show()

        text_data[text_index] = df

    return text_data


def compute_error_for_seven_points_homography():
    calibration_data = ReadData.read_calibration_data()

    # visualize the manual calibration data after icp calibration.
    avg_distance_list = []
    for subject_index in range(0, 19):
        calibration_gaze_list = calibration_data[subject_index][0]
        calibration_avg_list = calibration_data[subject_index][1]
        calibration_point_list = calibration_data[subject_index][2]
        calibration_data[subject_index][0] = [[calibration_gaze_list[0][0], calibration_gaze_list[0][14], calibration_gaze_list[0][29]],
                                              [calibration_gaze_list[2][15]],
                                              [calibration_gaze_list[5][0], calibration_gaze_list[5][14], calibration_gaze_list[5][29]]]
        calibration_data[subject_index][1] = [[calibration_avg_list[0][0], calibration_avg_list[0][14], calibration_avg_list[0][29]],
                                              [calibration_avg_list[2][15]],
                                              [calibration_avg_list[5][0], calibration_avg_list[5][14], calibration_avg_list[5][29]]]
        calibration_data[subject_index][2] = [[calibration_point_list[0][0], calibration_point_list[0][14], calibration_point_list[0][29]],
                                              [calibration_point_list[2][15]],
                                              [calibration_point_list[5][0], calibration_point_list[5][14], calibration_point_list[5][29]]]

        transform_matrix = ManualCalibrateForStd.compute_std_cali_with_homography_matrix(subject_index, calibration_data)

        gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list, \
            avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list, \
            calibration_point_list_modified = ManualCalibrateForStd.apply_transform_to_calibration(subject_index, calibration_data, transform_matrix)

        distance_list, avg_distance = ManualCalibrateForStd.compute_distance_between_std_and_correction(avg_gaze_coordinate_after_translation_list, calibration_point_list_modified)
        print("avg_distance: ", avg_distance)
        avg_distance_list.append(avg_distance)
        # Render.visualize_cali_result(gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list,
        #                              avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list,
        #                              calibration_point_list_modified)


def save_text_density_for_main(reading_data_after_trim, text_data, calibration_data):
    # # compute the text density for each subject and save it.
    text_density_info_list = ComputeTextDensity.compute_text_density(reading_data_after_trim, text_data, calibration_data)
    SaveFiles.save_text_density(text_density_info_list)


def update_reading_data_and_save(reading_data, text_data, calibration_data):
    # # add density for each reading_data
    reading_data = UtilFunctions.compute_density_for_reading_data(reading_data)

    reading_data_after_transform, reading_data_after_trim, reading_data_after_restore = UtilFunctions.trim_data(reading_data, calibration_data)
    SaveFiles.save_reading_data_after_modification(reading_data_after_restore, "reading_after_trim")


def check_y_distribution_of_data_given_row_label(reading_data):
    for subject_index in range(len(reading_data)):
        gaze_data_by_row = [[] for _ in range(configs.row_num)]
        for text_index in range(len(reading_data[subject_index])):
            df = reading_data[subject_index][text_index]
            row_label = df["row_label"].tolist()
            gaze_y = df["gaze_y"].tolist()
            for gaze_index in range(len(row_label)):
                row = row_label[gaze_index]
                gaze_data_by_row[row].append(gaze_y[gaze_index])

        fig = plt.figure(figsize=(12, 8))
        ax_list = []
        for i in range(1, 7):
            ax = fig.add_subplot(2, 3, i)
            ax_list.append(ax)

        for row in range(configs.row_num):
            ax_list[row].hist(gaze_data_by_row[row], bins=100, density=True, alpha=0.75)
            ax_list[row].set_title(f"row: {row}")
            ax_list[row].set_ylim(0, 0.2)
            ax_list[row].set_aspect('auto', adjustable='box')
            print("gaze point number of given row: ", len(gaze_data_by_row[row]))
        print()

        plt.show()

