import os.path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

import ManualCalibrateForStd
import UtilFunctions
import configs


def visualize_all_text_and_gaze_point(text_data, text_coordinate_list, text_row_col_list, gaze_coordinate_list_after_transform_1, gaze_col_row_label_list_1, calibration_avg_gaze_list_1d_after_transform):
    fig = plt.figure(figsize=(24, 16))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_aspect('equal', adjustable='box')

    text_coordinate_x_list = []
    text_coordinate_y_list = []
    text_color_list = []
    gaze_coordinate_x_list = []
    gaze_coordinate_y_list = []
    gaze_color_list = []

    # row_colors = plt.cm.get_cmap('viridis', configs.row_num)
    # column_colors = plt.cm.get_cmap('plasma', configs.col_num)
    # row_colors = row_colors.colors[:, 0:3]
    # column_colors = column_colors.colors[:, 0:3]

    color_1 = [0.3, 0.8, 0.8]
    color_2 = [0.8, 0.3, 0.8]
    color_3 = [0.8, 0.8, 0.3]
    color_4 = [0.3, 0.3, 0.8]

    for text_index in range(len(text_data)):
        for text_unit_index in range(len(text_coordinate_list[text_index])):
            text_coordinate = text_coordinate_list[text_index][text_unit_index]
            row = int(text_row_col_list[text_index][text_unit_index][0])
            col = int(text_row_col_list[text_index][text_unit_index][1])
            if (col + row) % 4 == 0:
                color = color_1
            elif (col + row) % 3 == 0:
                color = color_2
            elif (col + row) % 2 == 0:
                color = color_3
            else:
                color = color_4
            # ax.scatter(text_coordinate[0], text_coordinate[1], c=color, marker='o', s=10)
            text_coordinate_x_list.append(text_coordinate[0])
            text_coordinate_y_list.append(text_coordinate[1])
            text_color_list.append(color)

        for gaze_index in range(len(gaze_coordinate_list_after_transform_1[text_index])):
            gaze_coordinate = gaze_coordinate_list_after_transform_1[text_index][gaze_index]
            row = int(gaze_col_row_label_list_1[text_index][gaze_index][0])
            col = int(gaze_col_row_label_list_1[text_index][gaze_index][1])
            if (col + row) % 4 == 0:
                color = color_1
            elif (col + row) % 3 == 0:
                color = color_2
            elif (col + row) % 2 == 0:
                color = color_3
            else:
                color = color_4
            # ax.scatter(gaze_coordinate[0], gaze_coordinate[1], c=color, marker='x', s=1)
            gaze_coordinate_x_list.append(gaze_coordinate[0])
            gaze_coordinate_y_list.append(gaze_coordinate[1])
            gaze_color_list.append(color)

    ax.scatter(text_coordinate_x_list, text_coordinate_y_list, c='k', marker='o', s=10, zorder=5)
    ax.scatter(gaze_coordinate_x_list, gaze_coordinate_y_list, c=gaze_color_list, marker='x', s=1)
    ax.scatter(calibration_avg_gaze_list_1d_after_transform[:, :, 0], calibration_avg_gaze_list_1d_after_transform[:, :, 1], c='k', marker='^', s=15, zorder=10)
    ax.scatter(calibration_avg_gaze_list_1d_after_transform[:, :, 0], calibration_avg_gaze_list_1d_after_transform[:, :, 1], c='red', marker='^', s=1, zorder=11)
    plt.show()


def visualize_single_text_unit_and_gaze_point(subject_index, text_data, text_coordinate_list, text_row_col_list, gaze_coordinate_list_after_transform_1, gaze_col_row_label_list_1, calibration_avg_gaze_list_1d_after_transform):
    gaze_list_for_visualization = [[[] for _ in range(configs.col_num)] for _ in range(configs.row_num)]
    avg_calibration_list_for_visualization = [[[] for _ in range(configs.col_num)] for _ in range(configs.row_num)]
    text_list_for_visualization = [[[] for _ in range(configs.col_num)] for _ in range(configs.row_num)]

    for text_index in range(len(text_row_col_list)):
        for row_col_index in range(len(text_row_col_list[text_index])):
            text_coordinate = text_coordinate_list[text_index][row_col_index]
            row = int(text_row_col_list[text_index][row_col_index][0])
            col = int(text_row_col_list[text_index][row_col_index][1])
            avg_calibration = calibration_avg_gaze_list_1d_after_transform[row][col]
            gaze_coordinates = []
            for gaze_index in range(len(gaze_col_row_label_list_1[text_index])):
                gaze_row = int(gaze_col_row_label_list_1[text_index][gaze_index][0])
                gaze_col = int(gaze_col_row_label_list_1[text_index][gaze_index][1])
                if gaze_row == row and gaze_col == col:
                    gaze_coordinates.append(gaze_coordinate_list_after_transform_1[text_index][gaze_index])

            gaze_list_for_visualization[row][col].extend(np.array(gaze_coordinates))
            avg_calibration_list_for_visualization[row][col].append(avg_calibration)
            text_list_for_visualization[row][col].append(text_coordinate)

    for row_index in range(configs.row_num):
        for col_index in range(configs.col_num):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)

            gaze_coordinates = np.array(gaze_list_for_visualization[row_index][col_index])
            avg_calibration = avg_calibration_list_for_visualization[row_index][col_index][0]
            text_coordinate = text_list_for_visualization[row_index][col_index][0]

            ax.set_xlim(text_coordinate[0] - configs.text_height * 1.5, text_coordinate[0] + configs.text_height * 1.5)
            ax.set_ylim(text_coordinate[1] + configs.text_height * 1.5, text_coordinate[1] - configs.text_height * 1.5)
            ax.set_aspect('equal', adjustable='box')

            ax.scatter(text_coordinate[0], text_coordinate[1], c='k', marker='o', s=10, zorder=5)
            if len(gaze_coordinates) > 0:
                ax.scatter(gaze_coordinates[:, 0], gaze_coordinates[:, 1], c='g', marker='x', s=1)
            ax.scatter(avg_calibration[0], avg_calibration[1], c='r', marker='^', s=15, zorder=10)
            save_path = f"pic/cluster_of_reading_and_cali/subject_{subject_index}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(f"{save_path}/row_{row_index}_col_{col_index}.png")


def cluster_reading_data(reading_data, text_data, calibration_data):
    # 基于标准文字位置来创建nbrs。
    text_nbrs_list = []
    text_coordinate_list = []
    text_row_col_list = []
    for text_index in range(len(text_data)):
        text_df = text_data[text_index]
        text_df = text_df[text_df["word"] != "blank_supplement"]
        text_coordinates = text_df[["x", "y"]].values.tolist()
        text_coordinate_list.append(text_coordinates)
        text_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(text_coordinates)
        text_nbrs_list.append(text_nbrs)
        text_row_col = text_df[["row", "col"]].values.tolist()
        text_row_col_list.append(text_row_col)

    gaze_col_row_label_list = []

    for subject_index in range(len(reading_data)):
        print(f"current subject: {subject_index}")
        transform_matrix = ManualCalibrateForStd.compute_std_cali_with_homography_matrix(subject_index, calibration_data)
        calibration_avg_gaze_list = calibration_data[subject_index][1]
        calibration_avg_gaze_list_1d = []

        for row_index in range(len(calibration_avg_gaze_list)):
            for col_index in range(len(calibration_avg_gaze_list[row_index])):
                calibration_avg_gaze_list_1d.append([calibration_avg_gaze_list[row_index][col_index]["avg_gaze_x"], calibration_avg_gaze_list[row_index][col_index]["avg_gaze_y"]])

        calibration_avg_gaze_list_1d_homogeneous = [UtilFunctions.change_2d_vector_to_homogeneous_vector(calibration_avg_gaze) for calibration_avg_gaze in calibration_avg_gaze_list_1d]
        calibration_avg_gaze_list_1d_homogeneous = [np.dot(transform_matrix, calibration_avg_gaze) for calibration_avg_gaze in calibration_avg_gaze_list_1d_homogeneous]
        calibration_avg_gaze_list_1d_after_transform = [UtilFunctions.change_homogeneous_vector_to_2d_vector(calibration_avg_gaze) for calibration_avg_gaze in calibration_avg_gaze_list_1d_homogeneous]
        calibration_avg_gaze_list_1d_after_transform = np.array(calibration_avg_gaze_list_1d_after_transform)
        calibration_avg_gaze_list_1d_after_transform = calibration_avg_gaze_list_1d_after_transform.reshape((configs.row_num, configs.col_num, 2))

        # 基于移动后的校准点来创建nbrs。
        calibration_avg_gaze_list_of_each_text = []
        calibration_avg_nrbs_list = []
        for text_index in range(len(text_data)):
            text_df = text_data[text_index]
            text_df = text_df[text_df["word"] != "blank_supplement"]
            text_row_col = text_df[["row", "col"]].values.tolist()
            calibration_avg_gaze_coordinates = []
            for row_col_index in range(len(text_row_col)):
                row = int(text_row_col[row_col_index][0])
                col = int(text_row_col[row_col_index][1])
                calibration_avg_gaze_coordinates.append(calibration_avg_gaze_list_1d_after_transform[row][col])
            calibration_avg_gaze_list_of_each_text.append(calibration_avg_gaze_coordinates)
            calibration_avg_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(calibration_avg_gaze_coordinates)
            calibration_avg_nrbs_list.append(calibration_avg_nbrs)

        gaze_col_row_label_list_1 = []
        gaze_coordinate_list_after_transform_1 = []

        for text_index in range(len(text_data)):
            reading_df = reading_data[subject_index][text_index]
            gaze_coordinates = reading_df[["gaze_x", "gaze_y"]].values.tolist()
            gaze_coordinates_homogeneous = [UtilFunctions.change_2d_vector_to_homogeneous_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates]
            gaze_coordinates_homogeneous = [np.dot(transform_matrix, gaze_coordinate) for gaze_coordinate in gaze_coordinates_homogeneous]
            gaze_coordinate_list_after_transform_2 = [UtilFunctions.change_homogeneous_vector_to_2d_vector(gaze_coordinate) for gaze_coordinate in gaze_coordinates_homogeneous]

            # distances, indices = text_nbrs_list[text_index].kneighbors(gaze_coordinate_list_after_transform_2)
            distances, indices = calibration_avg_nrbs_list[text_index].kneighbors(gaze_coordinate_list_after_transform_2)
            gaze_col_row_label_list_2 = []
            for gaze_index, index in enumerate(indices):
                row = text_row_col_list[text_index][index[0]][0]
                col = text_row_col_list[text_index][index[0]][1]
                gaze_col_row_label_list_2.append([row, col])

            gaze_col_row_label_list_1.append(gaze_col_row_label_list_2)
            gaze_coordinate_list_after_transform_1.append(gaze_coordinate_list_after_transform_2)

        # 将所有的gaze点和text点都绘制出来。
        visualize_all_text_and_gaze_point(text_data, text_coordinate_list, text_row_col_list, gaze_coordinate_list_after_transform_1, gaze_col_row_label_list_1, calibration_avg_gaze_list_1d_after_transform)

        # 将单个text点对应的avg_calibration和gaze点绘制出来。
        # visualize_single_text_unit_and_gaze_point(subject_index, text_data, text_coordinate_list, text_row_col_list, gaze_coordinate_list_after_transform_1, gaze_col_row_label_list_1, calibration_avg_gaze_list_1d_after_transform)


