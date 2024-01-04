import os

import pandas as pd

import configs


def save_text_density(text_density_info_list):
    gaze_data_file_path = f"modified_gaze_data/{configs.round_num}/tobii"
    subject_name_list = os.listdir(gaze_data_file_path)

    for subject_index in range(len(text_density_info_list)):
        file_path = f"text_density/{configs.round_num}/tobii/{subject_name_list[subject_index]}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # concatenate the dataframe in text_density_info_list[subject_index]
        text_density_df = pd.concat(text_density_info_list[subject_index], ignore_index=True)
        text_density_df.to_csv(f"{file_path}/text_density.csv", index=False, encoding="utf-8-sig")


def save_reading_data_after_modification(reading_data_after_modification, name):
    file_path_prefix = f"original_gaze_data/{configs.round_num}/tobii"
    subject_name_list = os.listdir(file_path_prefix)
    for subject_index in range(len(subject_name_list)):
        file_path = f"{file_path_prefix}/{subject_name_list[subject_index]}/{name}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        for text_index in range(len(reading_data_after_modification[subject_index])):
            reading_df = reading_data_after_modification[subject_index][text_index]
            reading_df.to_csv(f"{file_path}/{text_index}.csv", index=False, encoding="utf-8-sig")


def save_text_data_after_adding_boundary_and_penalty(text_data):
    file_path_prefix = f"text/{configs.round_num}/text_sorted_mapping_with_boundary_and_penalty.csv"
    df = pd.concat(text_data, ignore_index=True)
    df.to_csv(file_path_prefix, index=False, encoding="utf-8-sig")


