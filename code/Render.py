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


def visualize_error_of_reading_matching(subject_index, file_prefix, ax=None):
    file_path = f"{file_prefix}/subject_{subject_index}.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split(",") for line in lines]
        line_dict_list = []
        for index, line in enumerate(lines):
            line_avg_error = float(line[0].split(":")[1])
            line_last_iteration = float(line[1].split(":")[1])
            line_dict_list.append({"line_avg_error": line_avg_error, "line_last_iteration": line_last_iteration})

        if not ax:
            fig = plt.figure(figsize=(24, 12))
            ax = fig.add_subplot(111)

        ax.scatter([i for i in range(len(line_dict_list))], [line_dict["line_avg_error"] for line_dict in line_dict_list], c='g', marker='x', s=1)
        ax.plot([i for i in range(len(line_dict_list))], [line_dict["line_avg_error"] for line_dict in line_dict_list], c='g')
        ax_twin = ax.twinx()
        ax_twin.scatter([i for i in range(len(line_dict_list))], [line_dict["line_last_iteration"] for line_dict in line_dict_list], c='r', marker='x', s=1)
        ax_twin.plot([i for i in range(len(line_dict_list))], [line_dict["line_last_iteration"] for line_dict in line_dict_list], c='r')

        # add text
        for i in range(len(line_dict_list)):
            ax.text(i, line_dict_list[i]["line_avg_error"], f"{line_dict_list[i]['line_avg_error']:.2f}", fontsize=5)
            ax_twin.text(i, line_dict_list[i]["line_last_iteration"], f"{line_dict_list[i]['line_last_iteration']:.2f}", fontsize=5)

        ax.set_xlabel("line index")
        ax.set_ylabel("avg error (green)")
        ax_twin.set_ylabel("last iteration (red)")
        ax.set_title(f"subject {subject_index}")
        plt.show()
