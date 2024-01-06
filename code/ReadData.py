import os
import pandas as pd
import configs


def read_gaze_data(data_type, reading_type) -> list:
    # Read data
    data_path_prefix = f"{data_type}_gaze_data/{configs.round_num}/tobii/"
    subject_path_list = os.listdir(data_path_prefix)
    subject_path_list.sort()

    subject_list = []
    for subject_index, subject_name in enumerate(subject_path_list):
        subject_path = f"{data_path_prefix}/{subject_name}/{reading_type}/"
        reading_path_list = os.listdir(subject_path)
        reading_path_list.sort(key=lambda x: int(x.split(".")[0]))
        reading_file_list = []
        for reading_index, reading_name in enumerate(reading_path_list):
            file_path = f"{subject_path}/{reading_name}"
            pd_reading_file = pd.read_csv(file_path)
            reading_file_list.append(pd_reading_file)
        subject_list.append(reading_file_list)

    return subject_list


def read_text_data(file_name) -> list:
    data_path_prefix = f"text/{configs.round_num}/{file_name}"
    pd_text_file = pd.read_csv(data_path_prefix)
    if "Unnamed: 0" in pd_text_file.columns:
        pd_text_file.drop(columns=["Unnamed: 0"], inplace=True)
    # divide pd_text_file according to its para_id
    para_id_list = pd_text_file["para_id"].unique()
    para_id_list.sort()
    pd_text_file_list = []
    for para_id in para_id_list:
        pd_text_file_list.append(pd_text_file[pd_text_file["para_id"] == para_id])

    return pd_text_file_list


def read_calibration_data() -> list:
    '''

    :return:
    subject_list: within each subject, there are 3 lists
        calibration data list: 1st layer -> subject, 2nd layer -> calibration points, 3rd layer -> all calibration data [x_1, y_1], [x_2, y_2], ...
        calibration avg data list: 1st layer -> subject, 2nd layer -> calibration points, 3rd layer -> avg calibration data [avg_x, avg_y]
        calibration point list: 1st layer -> calibration points, 2nd layer -> [x, y]
    '''
    data_path_prefix = f"original_gaze_data/{configs.round_num}/tobii/"
    subject_path_list = os.listdir(data_path_prefix)
    subject_path_list.sort()

    subject_list = []
    for subject_index, subject_name in enumerate(subject_path_list):
        subject_path = f"{data_path_prefix}/{subject_name}/calibration.csv"
        pd_calibration_file = pd.read_csv(subject_path)

        matrix_x_uniques = pd_calibration_file["matrix_x"].unique()
        matrix_x_uniques.sort()
        matrix_y_uniques = pd_calibration_file["matrix_y"].unique()
        matrix_y_uniques.sort()
        gaze_list = [[None for _ in range(len(matrix_x_uniques))] for _ in range(len(matrix_y_uniques))]
        for matrix_y_index, matrix_y in enumerate(matrix_y_uniques):
            for matrix_x_index, matrix_x in enumerate(matrix_x_uniques):
                calibration_gaze_x = pd_calibration_file[(pd_calibration_file["matrix_x"] == matrix_x) & (pd_calibration_file["matrix_y"] == matrix_y)]
                calibration_gaze_x = calibration_gaze_x["gaze_x"].tolist()[1:]
                calibration_gaze_x = [x for x in calibration_gaze_x if x != "failed"]
                calibration_gaze_x = [float(x) for x in calibration_gaze_x]
                calibration_gaze_y = pd_calibration_file[(pd_calibration_file["matrix_x"] == matrix_x) & (pd_calibration_file["matrix_y"] == matrix_y)]
                calibration_gaze_y = calibration_gaze_y["gaze_y"].tolist()[1:]
                calibration_gaze_y = [y for y in calibration_gaze_y if y != "failed"]
                calibration_gaze_y = [float(y) for y in calibration_gaze_y]
                calibration = {"gaze_x": calibration_gaze_x, "gaze_y": calibration_gaze_y}
                gaze_list[matrix_y_index][matrix_x_index] = calibration

        avg_gaze_list = [[None for _ in range(len(matrix_x_uniques))] for _ in range(len(matrix_y_uniques))]
        for matrix_y_index, matrix_y in enumerate(matrix_y_uniques):
            for matrix_x_index, matrix_x in enumerate(matrix_x_uniques):
                avg_x = sum(gaze_list[matrix_y_index][matrix_x_index]["gaze_x"]) / len(gaze_list[matrix_y_index][matrix_x_index]["gaze_x"])
                avg_y = sum(gaze_list[matrix_y_index][matrix_x_index]["gaze_y"]) / len(gaze_list[matrix_y_index][matrix_x_index]["gaze_y"])
                avg_gaze_list[matrix_y_index][matrix_x_index] = {"avg_gaze_x": avg_x, "avg_gaze_y": avg_y}

        calibration_point_list = [[None for _ in range(len(matrix_x_uniques))] for _ in range(len(matrix_y_uniques))]
        for matrix_y_index, matrix_y in enumerate(matrix_y_uniques):
            for matrix_x_index, matrix_x in enumerate(matrix_x_uniques):
                point_x = 380 + 40 * matrix_x
                point_y = 272 + 64 * matrix_y
                calibration_point_list[matrix_y_index][matrix_x_index] = {"point_x": point_x, "point_y": point_y}

        subject_list.append([gaze_list, avg_gaze_list, calibration_point_list])

    return subject_list


