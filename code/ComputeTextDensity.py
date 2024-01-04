from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors


def compute_text_density(reading_data, text_data, calibration_data):
    text_coordinates_list = []
    text_info_list = []
    nbrs_list = []
    for text_index in range(len(text_data)):
        text_df = text_data[text_index]
        text_x = text_df["x"].tolist()
        text_y = text_df["y"].tolist()
        row_list = text_df["row"].tolist()
        col_list = text_df["col"].tolist()
        word_list = text_df["word"].tolist()

        text_coordinates = [[text_x[i], text_y[i]] for i in range(len(text_x))]
        text_coordinates_list.append(text_coordinates)

        text_info = [{"row": row_list[i], "col": col_list[i], "word": word_list[i]} for i in range(len(row_list))]
        text_info_list.append(text_info)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(text_coordinates)
        nbrs_list.append(nbrs)

    text_density_list_1 = [[] for _ in range(len(reading_data))]
    for subject_index in range(len(reading_data)):
        text_density_list_2 = [[] for _ in range(len(reading_data[subject_index]))]
        for text_index in range(len(reading_data[subject_index])):
            reading_df = reading_data[subject_index][text_index]
            text_density_list_3 = [0 for _ in range(len(text_coordinates_list[text_index]))]
            for index, row in reading_df.iterrows():
                reading_x = row["gaze_x"]
                reading_y = row["gaze_y"]
                reading_point = [[reading_x, reading_y]]
                distances, indices = nbrs_list[text_index].kneighbors(reading_point)

                text_density_list_3[indices[0][0]] += 1
            # # visualize to gurantee the result.
            # fig = plt.figure(figsize=(12, 8))
            # ax = fig.add_subplot(111)
            # ax.set_xlim(0, 1920)
            # ax.set_ylim(1080, 0)
            # ax.set_aspect('equal', adjustable='box')
            # for text_unit_index in range(len(text_coordinates_list[text_index])):
            #     text_density = text_density_list_3[text_unit_index]
            #     color_ratio = 0.7 - text_density / max(text_density_list_3) * 0.7
            #     color = [color_ratio, color_ratio, color_ratio]
            #     ax.scatter(text_coordinates_list[text_index][text_unit_index][0], text_coordinates_list[text_index][text_unit_index][1], c=color, marker='o', s=10)
            # for index, row in reading_df.iterrows():
            #     reading_x = row["gaze_x"]
            #     reading_y = row["gaze_y"]
            #     ax.scatter(reading_x, reading_y, c=[1, 0.8, 0.8], marker='x', s=1)
            # plt.show()

            text_density_list_2[text_index] = text_density_list_3
        text_density_list_1[subject_index] = text_density_list_2

    text_density_info_list = []
    for subject_index in range(len(text_density_list_1)):
        total_gaze_points = 0
        text_density_info_list_1 = []
        for text_index in range(len(text_density_list_1[subject_index])):
            text_density_df = text_data[text_index].copy()
            text_density_df["text_density"] = text_density_list_1[subject_index][text_index]
            total_gaze_points += sum(text_density_list_1[subject_index][text_index])
            text_density_info_list_1.append(text_density_df)

        for text_index in range(len(text_density_list_1[subject_index])):
            text_density_df = text_density_info_list_1[text_index]
            text_density_df["relative_text_density"] = text_density_list_1[subject_index][text_index]
            text_density_df["relative_text_density"] /= total_gaze_points
            text_density_info_list_1[text_index] = text_density_df

        text_density_info_list.append(text_density_info_list_1)

    return text_density_info_list
