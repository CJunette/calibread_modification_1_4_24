import numpy as np
from matplotlib import pyplot as plt

import ManualCalibrateForStd
import UtilFunctions


def render_text_and_reading(text_data, reading_data):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, -500)

    # 字宽40，高64。中心点坐标为text_data["x"], text_data["y"]
    # 读取reading_data中的数据，绘制矩形框
    # 读取text_data中的数据，绘制文本
    for index, row in text_data.iterrows():
        if row["word"] == " ":
            continue
        if row["word"] == "blank_supplement":
            ax.text(row["x"], row["y"], "·", fontsize=20)
        else:
            ax.text(row["x"], row["y"], row["word"], fontsize=20)

    for index, row in reading_data.iterrows():
        x = row["gaze_x"]
        y = row["gaze_y"]
        ax.scatter(x, y, s=10, c="blue")

    plt.show()


def visualize_cali_result(gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list,
                          avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list,
                          calibration_point_list_modified, bool_verbose=True, file_name=None):
    fig = plt.figure(figsize=(24, 16))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.scatter(calibration_point_list_modified[:, 0], calibration_point_list_modified[:, 1], c='k', marker='o', s=20)
    if bool_verbose:
        ax.scatter(gaze_coordinates_after_translation_list[:, 0], gaze_coordinates_after_translation_list[:, 1], c='g', marker='x', s=3)
        ax.scatter(gaze_coordinates_before_translation_list[:, 0], gaze_coordinates_before_translation_list[:, 1], c='b', marker='x', s=3)
    ax.scatter(avg_gaze_coordinate_after_translation_list[:, 0], avg_gaze_coordinate_after_translation_list[:, 1], c='orange', marker='^', s=6)
    ax.scatter(avg_gaze_coordinate_before_translation_list[:, 0], avg_gaze_coordinate_before_translation_list[:, 1], c='red', marker='^', s=6)

    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()

    plt.clf()
    plt.close()


def visualize_reading_data_after_process(reading_data, reading_data_after_process, calibration_data, subject_index):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(-500, 2500)
    ax.set_ylim(1500, -500)
    ax.set_aspect('equal', adjustable='box')

    calibration_point_list = calibration_data[subject_index][2]
    for row_index in range(len(calibration_point_list)):
        for col_index in range(len(calibration_point_list[row_index])):
            calibration_point_dict = calibration_point_list[row_index][col_index]
            calibration_point_x = calibration_point_dict["point_x"]
            calibration_point_y = calibration_point_dict["point_y"]
            calibration_point = np.array([calibration_point_x, calibration_point_y])
            ax.scatter(calibration_point[0], calibration_point[1], c='k', marker='o', s=10)

    for text_index in range(len(reading_data)):
        reading_df = reading_data[text_index]
        reading_gaze_x = reading_df["gaze_x"].tolist()
        reading_gaze_y = reading_df["gaze_y"].tolist()
        ax.scatter(reading_gaze_x, reading_gaze_y, c='g', marker='x', s=1)

        reading_df_after_trim = reading_data_after_process[text_index]
        reading_gaze_x_after_process = reading_df_after_trim["gaze_x"].tolist()
        reading_gaze_y_after_process = reading_df_after_trim["gaze_y"].tolist()
        ax.scatter(reading_gaze_x_after_process, reading_gaze_y_after_process, c=(0.4, 0.3, 0.75), marker='x', s=1)

    plt.title(f"subject_index: {subject_index}")
    plt.show()


def visualize_text_data(text_df, color='k', size=10):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(-500, 2500)
    ax.set_ylim(1500, -500)
    ax.set_aspect('equal', adjustable='box')

    for index, row in text_df.iterrows():
        x = row["x"]
        y = row["y"]
        ax.scatter(x, y, c=color, marker='o', s=size)

    plt.show()


def visualize_error_of_reading_matching(subject_index, file_prefix, ax=None):
    file_path = f"{file_prefix}/subject_{subject_index}.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split(",") for line in lines]
        line_dict_list = []
        for index, line in enumerate(lines[1: ]):
            line_avg_error = float(line[0].split(":")[1])
            line_last_iteration = float(line[1].split(":")[1])
            line_algorithm_error = float(line[2].split(":")[1])
            line_dict_list.append({"line_avg_error": line_avg_error, "line_last_iteration": line_last_iteration, "algorithm_error": line_algorithm_error})

        if not ax:
            fig = plt.figure(figsize=(24, 12))
            ax = fig.add_subplot(111)

        ax.scatter([i for i in range(len(line_dict_list))], [line_dict["line_avg_error"] for line_dict in line_dict_list], c='g', marker='x', s=1)
        ax.plot([i for i in range(len(line_dict_list))], [line_dict["line_avg_error"] for line_dict in line_dict_list], c='g')
        ax_twin_iteration = ax.twinx()
        ax_twin_iteration.scatter([i for i in range(len(line_dict_list))], [line_dict["line_last_iteration"] for line_dict in line_dict_list], c='r', marker='x', s=1)
        ax_twin_iteration.plot([i for i in range(len(line_dict_list))], [line_dict["line_last_iteration"] for line_dict in line_dict_list], c='r')
        ax_twin_algorithm_error = ax.twinx()
        ax_twin_algorithm_error.scatter([i for i in range(len(line_dict_list))], [line_dict["algorithm_error"] for line_dict in line_dict_list], c='b', marker='x', s=1)
        ax_twin_algorithm_error.plot([i for i in range(len(line_dict_list))], [line_dict["algorithm_error"] for line_dict in line_dict_list], c='b')

        # add text
        for i in range(len(line_dict_list)):
            ax.text(i, line_dict_list[i]["line_avg_error"], f"{line_dict_list[i]['line_avg_error']:.2f}", fontsize=5)
            ax_twin_iteration.text(i, line_dict_list[i]["line_last_iteration"], f"{line_dict_list[i]['line_last_iteration']:.2f}", fontsize=5)
            ax_twin_algorithm_error.text(i, line_dict_list[i]["algorithm_error"], f"{line_dict_list[i]['algorithm_error']:.2f}", fontsize=5)

        ax.set_xlabel("line index")
        ax.set_ylabel("avg error (green)")
        ax_twin_iteration.set_ylabel("last iteration (red)")
        ax_twin_algorithm_error.set_ylabel("algorithm error (blue)")
        ax.set_title(f"subject {subject_index}")
        plt.show()


def render_cali_points_and_avg_gaze(calibration_data):
    std_cali_points = calibration_data[0][2]

    fig = plt.figure(figsize=(24, 16))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_aspect('equal', adjustable='box')

    color_1 = [0.3, 0.8, 0.8]
    color_2 = [0.8, 0.3, 0.8]
    color_3 = [0.8, 0.8, 0.3]
    color_4 = [0.3, 0.3, 0.8]
    color_5 = [0.8, 0.3, 0.3]

    for row_index in range(len(std_cali_points)):
        for col_index in range(len(std_cali_points[row_index])):
            std_cali_point_dict = std_cali_points[row_index][col_index]
            std_cali_point_x = std_cali_point_dict["point_x"]
            std_cali_point_y = std_cali_point_dict["point_y"]
            std_cali_point = np.array([std_cali_point_x, std_cali_point_y])
            ax.scatter(std_cali_point[0], std_cali_point[1], c='k', marker='o', s=20, zorder=3)

    for subject_index in range(len(calibration_data)):
        if subject_index == 13:
            continue
        transform_matrix = ManualCalibrateForStd.compute_std_cali_with_homography_matrix(subject_index, calibration_data)
        avg_gaze_coordinates = calibration_data[subject_index][1]
        avg_gaze_coordinates = [[[avg_gaze_coordinates[i][j]["avg_gaze_x"], avg_gaze_coordinates[i][j]["avg_gaze_y"]] for j in range(len(avg_gaze_coordinates[i]))] for i in range(len(avg_gaze_coordinates))]
        avg_gaze_coordinates_homogeneous = [[UtilFunctions.change_2d_vector_to_homogeneous_vector(avg_gaze_coordinates[i][j]) for j in range(len(avg_gaze_coordinates[i]))] for i in range(len(avg_gaze_coordinates))]
        avg_gaze_coordinates_homogeneous_after_transform = [[np.dot(transform_matrix, avg_gaze_coordinates_homogeneous[i][j]) for j in range(len(avg_gaze_coordinates_homogeneous[i]))] for i in range(len(avg_gaze_coordinates_homogeneous))]
        avg_gaze_coordinates_after_transform = [[UtilFunctions.change_homogeneous_vector_to_2d_vector(avg_gaze_coordinates_homogeneous_after_transform[i][j]) for j in range(len(avg_gaze_coordinates_homogeneous_after_transform[i]))] for i in range(len(avg_gaze_coordinates_homogeneous_after_transform))]

        for row_index in range(len(avg_gaze_coordinates_after_transform)):
            for col_index in range(len(avg_gaze_coordinates_after_transform[row_index])):
                if (col_index + row_index) % 5 == 0:
                    color = color_5
                elif (col_index + row_index) % 4 == 0:
                    color = color_4
                elif (col_index + row_index) % 3 == 0:
                    color = color_3
                elif (col_index + row_index) % 2 == 0:
                    color = color_2
                else:
                    color = color_1

                avg_gaze_coordinate_after_transform = avg_gaze_coordinates_after_transform[row_index][col_index]
                avg_gaze_coordinate_x = avg_gaze_coordinate_after_transform[0]
                avg_gaze_coordinate_y = avg_gaze_coordinate_after_transform[1]
                ax.scatter(avg_gaze_coordinate_x, avg_gaze_coordinate_y, c=color, marker='^', s=10, zorder=2)

    plt.show()

