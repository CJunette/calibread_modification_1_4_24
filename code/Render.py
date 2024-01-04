import numpy as np
from matplotlib import pyplot as plt


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
        ax.text(row["x"], row["y"], row["word"], fontsize=20)

    for index, row in reading_data.iterrows():
        x = row["gaze_x"]
        y = row["gaze_y"]
        ax.scatter(x, y, s=10, c="blue")

    plt.show()


def visualize_cali_result(gaze_coordinates_before_translation_list, gaze_coordinates_after_translation_list,
                          avg_gaze_coordinate_before_translation_list, avg_gaze_coordinate_after_translation_list,
                          calibration_point_list_modified, bool_verbose=True):
    fig = plt.figure(figsize=(12, 8))
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
    plt.show()


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
