import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import configs


def compress_gaze_and_cluster(reading_data):
    compress_ratio = 1 / (configs.right_down_text_center[0] - configs.left_top_text_center[0]) * configs.text_width
    for subject_index in range(len(reading_data)):
        gaze_x_list = []
        gaze_y_list = []
        compressed_gaze_x_list = []
        compressed_gaze_y_list = []
        gaze_info_list = []
        for text_index in range(len(reading_data[subject_index])):
            reading_df = reading_data[subject_index][text_index]
            gaze_x = reading_df["gaze_x"].tolist()
            gaze_y = reading_df["gaze_y"].tolist()
            compressed_gaze_x = [gaze_x[i] * compress_ratio for i in range(len(gaze_x))]
            compressed_gaze_y = [gaze_y[i] for i in range(len(gaze_y))]
            gaze_info = [{"text_index": text_index, "gaze_index": i} for i in range(len(gaze_x))]

            gaze_x_list.extend(gaze_x)
            gaze_y_list.extend(gaze_y)
            compressed_gaze_x_list.extend(compressed_gaze_x)
            compressed_gaze_y_list.extend(compressed_gaze_y)
            gaze_info_list.extend(gaze_info)

        # cluster compressed_gaze
        compressed_coordinates = [[compressed_gaze_x_list[i], compressed_gaze_y_list[i]] for i in range(len(compressed_gaze_x_list))]
        compressed_coordinates = np.array(compressed_coordinates)
        kmeans = KMeans(n_clusters=configs.row_num, random_state=configs.random_seed).fit(compressed_coordinates)
        labels = kmeans.labels_

        for text_index in range(len(reading_data[subject_index])):
            reading_data[subject_index][text_index]["row_label"] = [-1 for _ in range(reading_data[subject_index][text_index].shape[0])]

        for label_index in range(len(labels)):
            text_index = gaze_info_list[label_index]["text_index"]
            gaze_index = gaze_info_list[label_index]["gaze_index"]
            reading_data[subject_index][text_index]["row_label"].iloc[gaze_index] = labels[label_index]

        # # visualize.
        # compressed_gaze_x_list = compressed_coordinates[:, 0].tolist()
        # compressed_gaze_y_list = compressed_coordinates[:, 1].tolist()
        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(111)
        # ax.set_xlim(0, 1920)
        # ax.set_ylim(1080, 0)
        # ax.set_aspect('equal', adjustable='box')
        # ax.scatter(gaze_x_list, gaze_y_list, c='g', marker='o', s=1)
        # # visualize compressed coordinates with different colors
        # color_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
        #               [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        # for i in range(len(compressed_gaze_x_list)):
        #     ax.scatter(compressed_gaze_x_list[i], compressed_gaze_y_list[i], c=color_list[labels[i]], marker='x', s=1)
        # file_path = f"pic/cluster_gaze"
        # if not os.path.exists(file_path):
        #     os.makedirs(file_path)
        # plt.savefig(f"{file_path}/{subject_index}.jpeg", dpi=300)

    return reading_data
