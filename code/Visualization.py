import torch
from matplotlib import pyplot as plt

import Embeddings
import ReadData
import Render
import UtilFunctions
import configs


def visualize_single_gaze_trace_on_text():
    reading_data = ReadData.read_gaze_data("original", "reading")
    calibration_data = ReadData.read_calibration_data()
    text_data = ReadData.read_text_data("text_sorted_mapping.csv")

    # trim reading data to eliminate error points on left top corner (caused by exp errors). 已保存新数据，之后直接读取，无需调用。
    reading_data_after_transform, reading_data_after_trim, reading_data_after_restore = UtilFunctions.trim_data(reading_data, calibration_data)
    print("data trimming finished.")

    # visualize the reading data and text.
    for subject_index in range(0, 19):
        for text_index in range(30, 40):
            print(text_index)
            df = reading_data_after_trim[subject_index][text_index].iloc[:]
            Render.render_text_and_reading(text_data[text_index], df, df.shape[0] - 1)


def visualized_overlay_gaze_trace():
    reading_data = ReadData.read_gaze_data("original", "reading")
    calibration_data = ReadData.read_calibration_data()
    text_data = ReadData.read_text_data("text_sorted_mapping.csv")

    # trim reading data to eliminate error points on left top corner (caused by exp errors). 已保存新数据，之后直接读取，无需调用。
    reading_data_after_transform, reading_data_after_trim, reading_data_after_restore = UtilFunctions.trim_data(reading_data, calibration_data)
    print("data trimming finished.")

    # visualize reading after process.
    for subject_index in range(0, 19):
        fig = plt.figure(figsize=(27, 9))
        ax = fig.add_subplot(111)
        ax.set_xlim(200, 1720)
        ax.set_ylim(722, 140)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        # clear the outline
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_position([0.05, 0.05, 0.9, 0.9])
        ax.patch.set_alpha(0)

        for row_index in range(len(calibration_data[subject_index][2])):
            for col_index in range(len(calibration_data[subject_index][2][row_index])):
                x = calibration_data[subject_index][2][row_index][col_index]["point_x"]
                y = calibration_data[subject_index][2][row_index][col_index]["point_y"]
                # print(x, y)
                # add a rectangle
                rect = plt.Rectangle((x - configs.text_width / 2, y - configs.text_height / 2), configs.text_width, configs.text_height, color=(0.7, 0.7, 0.7), fill=False, linewidth=1)
                ax.add_patch(rect)
                if x == configs.left_top_text_center[0] or x == configs.right_down_text_center[0]:
                    rect = plt.Rectangle((x - configs.text_width / 2, y - configs.text_height / 2), configs.text_width, configs.text_height, color='red', fill=False, linewidth=1, zorder=2)
                    ax.add_patch(rect)

        for text_index in range(len(reading_data_after_trim[subject_index])):
            df = reading_data_after_trim[subject_index][text_index]
            gaze_x_list = df["gaze_x"].tolist()
            gaze_y_list = df["gaze_y"].tolist()
            ax.scatter(gaze_x_list, gaze_y_list, c=(0.2, 0.5, 0.8), marker='o', s=5, zorder=3, alpha=0.7)

        plt.show()
        # Render.visualize_reading_data_after_process(reading_data_after_transform[subject_index], reading_data_after_trim[subject_index], calibration_data, subject_index)


def get_neural_network_prediction():
    torch.manual_seed(configs.random_seed)

    # 准备数据
    X_train, y_train, X_val, y_val, X_train_info, X_val_info, vector_list, density_list, text_data, density_data = Embeddings.prepare_data(bool_token_embedding=True)
    input_size = X_train.shape[1]

    # 创建模型
    model, optimizer, criterion, epoch_num = Embeddings.create_simple_linear_model(input_size)

    # 读取模型
    model = Embeddings.read_model(model, f"model/simple_linear_net/009.pth")

    # 获取model对于每个word的预测值。
    model_result = Embeddings.return_model_prediction(model, X_train, X_val, X_train_info, X_val_info, text_data)

    for text_index in range(len(text_data)):
        prediction_list = []
        for index in range(text_data[text_index].shape[0]):
            row_index = text_data[text_index].iloc[index]["row"]
            col_index = text_data[text_index].iloc[index]["col"]
            prediction = model_result[text_index][row_index][col_index]
            prediction_list.append(prediction)
        text_data[text_index]["prediction"] = prediction_list

    Embeddings.visualize_linear_neural_network_prediction_using_row_unit(vector_list, density_data, text_data, model_result, save_index_str="011")


if __name__ == '__main__':
    # visualize_single_gaze_trace_on_text()
    # visualized_overlay_gaze_trace()

    get_neural_network_prediction()