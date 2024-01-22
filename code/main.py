import AddWeight
import CalibrateForReading
import ClusterGaze
import ClusterReadingData
import ComputeTextDensity
import Embeddings
import ICP
import ManualCalibrateForStd
import ReadData
import Render
import SaveFiles
import SplitText
import UtilFunctions


if __name__ == '__main__':
    # reading_data = ReadData.read_gaze_data("original", "reading")
    calibration_data = ReadData.read_calibration_data()
    # Render.render_cali_points_and_avg_gaze(calibration_data)

    # # add density for each reading_data. 已保存新数据（这里的保存是和trim一起完成的，没有单独写），之后直接读取，无需调用。
    # reading_data = UtilFunctions.compute_density_for_reading_data(reading_data)

    # trim reading data to eliminate error points on left top corner (caused by exp errors). 已保存新数据，之后直接读取，无需调用。
    # reading_data_after_transform, reading_data_after_trim, reading_data_after_restore = UtilFunctions.trim_data(reading_data, calibration_data)
    # print("data trimming finished.")
    # # visualize reading after process.
    # for subject_index in range(0, 19):
    #     Render.visualize_reading_data_after_process(reading_data_after_transform[subject_index], reading_data_after_trim[subject_index], calibration_data, subject_index)
    # save the trimmed reading data. 已保存新数据，之后直接读取，无需调用。
    # SaveFiles.save_reading_data_after_modification(reading_data_after_restore, "reading_after_trim")

    # text_data = ReadData.read_text_data("text_sorted_mapping_with_boundary_and_penalty.csv")

    # # 尝试将reading data的x坐标压缩，然后做聚类。将聚类的标签保留在每个gaze_point上。
    # reading_data = ReadData.read_gaze_data("original", "reading_after_trim")
    # reading_data = ClusterGaze.compress_gaze_and_cluster(reading_data)
    # SaveFiles.save_reading_data_after_modification(reading_data, "reading_after_cluster")

    # # 读取text数据，然后将其按照标点进行分割。对于分割后的小句子，逐一获取它们的embedding。
    # SplitText.save_text_data_after_splitting()
    # # 对于split得到的unique sentence，逐一获取它们的embedding。
    # Embeddings.save_embedding_for_unique_sentence()
    # # 获取text中分词的embedding。
    # Embeddings.save_embedding_for_tokens()
    # # 根据embedding来进行聚类。
    # Embeddings.cluster_word_given_embedding()
    # # 对已有的embedding，重新保存text_sorted_mapping。
    # Embeddings.save_text_for_model()
    # # 根据embedding来生成简单的神经网络。
    # Embeddings.linear_neural_network("")
    # # 对神经网络给出的结果进行更简单的可视化。
    # Embeddings.visualize_linear_neural_network_prediction("010")
    # # 将神经网络生成的结果与被试原始的注视时间结果进行可视化。
    # Embeddings.visualize_gaze_density_of_model("008")

    # # 将simple linear model的预测结果作为权重，放到text data中。
    # text_data = ReadData.read_text_data("text_sorted_mapping.csv")
    # text_data = Embeddings.add_model_prediction_to_text_data(text_data)
    # # 将加入权重后的text_data进行保存。
    # SaveFiles.save_text_data_after_add_prediction(text_data)

    # 为文本数据添加边界点，并根据边界点计算panelty。
    # 这里的数据可能会经常需要做修改（尤其是penalty）。所以虽然保存了，但也可能需要反复调用。
    # text_data = ReadData.read_text_data("text_sorted_mapping.csv")
    text_data = ReadData.read_text_data("text_sorted_mapping_with_prediction.csv")
    # add boundary points to text data.
    text_data = UtilFunctions.add_boundary_points_to_text_data(text_data)
    # add penalty to text_data。
    text_data = AddWeight.add_weight_to_text(text_data)
    # save text_data after adding boundary and penalty.
    # SaveFiles.save_text_data_after_adding_boundary_and_penalty(text_data)

    # text_data = ReadData.read_text_data("text_sorted_mapping_with_boundary_and_penalty.csv")
    # text_data = ReadData.read_text_data("text_sorted_mapping_with_prediction.csv")
    reading_data = ReadData.read_gaze_data("original", "reading_after_cluster")

    # # 计算每个text单元的text density. 已保存新数据，之后直接读取，无需调用。
    # text_density_info_list = ComputeTextDensity.compute_text_density(reading_data_after_trim, text_data, calibration_data)
    # SaveFiles.save_text_density(text_density_info_list)

    # 确认相同label的gaze data在y方向上的分布。
    # UtilFunctions.check_y_distribution_of_data_given_row_label(reading_data)

    # 看一下reading data相对于calibration points的分布。
    # ClusterReadingData.cluster_reading_data(reading_data, text_data, calibration_data)

    # # visualize the reading data and text.
    # for subject_index in range(0, 19):
    #     for text_index in range(0, 40):
    #         print(text_index)
    #         # Render.render_text_and_reading(text_data[text_index], reading_data[subject_index][text_index])
    #         Render.render_text_and_reading(text_data[text_index], reading_data_after_trim[subject_index][text_index])

    # # visualize manual calibration data with icp
    # icp_avg_distance_list = UtilFunctions.visualize_manual_calibration(ManualCalibrateForStd.compute_std_cali_with_icp)
    # # visualize manual calibration data with gradient descent of rotation and translation.
    # gd_rot_and_trans_avg_distance_list = UtilFunctions.visualize_manual_calibration(ManualCalibrateForStd.compute_std_cali_with_rotation_and_translation_gradient_descent)
    # # compare the calibration error of icp and rot_trans_gradient descent.
    # UtilFunctions.compare_manual_calibration_errors(ManualCalibrateForStd.compute_std_cali_with_icp, ManualCalibrateForStd.compute_std_cali_with_rotation_and_translation_gradient_descent)

    # visualize manual calibration data with gradient descent of the whole matrix.
    # gd_whole_avg_distance_list = UtilFunctions.visualize_manual_calibration(ManualCalibrateForStd.compute_std_cali_with_whole_matrix_gradient_descent)
    # visualize manual calibration data with homography matrix.
    # homo_avg_distance_list = UtilFunctions.visualize_manual_calibration(ManualCalibrateForStd.compute_std_cali_with_homography_matrix)
    # compare the calibration error of whole_matrix_gradient descent and homo.
    # UtilFunctions.compare_manual_calibration_errors(ManualCalibrateForStd.compute_std_cali_with_whole_matrix_gradient_descent, ManualCalibrateForStd.compute_std_cali_with_homography_matrix)

    # # compute the error of 7 point homography calibration.
    # UtilFunctions.compute_error_for_seven_points_homography()

    # 对reading数据，使用梯度下降实现对齐。
    avg_error_list = []
    for subject_index in range(0, 2):
        print(subject_index)
        # CalibrateForReading.calibrate_reading_with_whole_matrix_gradient_descent(subject_index, reading_data[subject_index], text_data, calibration_data, mode="location")
        # CalibrateForReading.calibrate_reading_with_whole_matrix_gradient_descent(subject_index, reading_data[subject_index], text_data, calibration_data, mode="location_and_coverage")
        # CalibrateForReading.calibrate_reading_with_whole_matrix_gradient_descent(subject_index, reading_data[subject_index], text_data, calibration_data, mode="location_coverage_and_penalty")
        # avg_errors = CalibrateForReading.calibrate_reading_with_whole_matrix_gradient_descent(subject_index, reading_data[subject_index], text_data, calibration_data, mode="location_coverage_penalty_and_rowlabel")
        # avg_errors = CalibrateForReading.calibrate_reading_with_whole_matrix_gradient_descent(subject_index, reading_data[subject_index], text_data, calibration_data, mode="torch")
        avg_errors = CalibrateForReading.calibrate_reading_with_whole_matrix_gradient_descent(subject_index, reading_data[subject_index], text_data, calibration_data, mode="simple_linear_weight")
    #     avg_errors.sort()
    #     print(avg_errors[:5])
    #     avg_error_list.append(avg_errors[:5])
    # for subject_index in range(len(avg_error_list)):
    #     print(avg_error_list[subject_index])

    # 可视化基于文本对齐的log数据。
    # for subject_index in range(0, 19):
    #     Render.visualize_error_of_reading_matching(subject_index, "log/gradient_descent_avg_error")


