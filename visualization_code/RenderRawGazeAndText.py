# 获取当前文件的上层目录
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# 获取code文件夹的路径
code_dir = os.path.join(parent_dir, 'code')
# 将code文件夹的路径添加到sys.path中
sys.path.append(code_dir)
import ReadData
import UtilFunctions
import Render


def render_gaze_trace_on_text():
    reading_data = ReadData.read_gaze_data("original", "reading")
    calibration_data = ReadData.read_calibration_data()

    # trim reading data to eliminate error points on left top corner (caused by exp errors). 已保存新数据，之后直接读取，无需调用。
    reading_data_after_transform, reading_data_after_trim, reading_data_after_restore = UtilFunctions.trim_data(reading_data, calibration_data)
    print("data trimming finished.")
    # visualize reading after process.
    for subject_index in range(0, 19):
        Render.visualize_reading_data_after_process(reading_data_after_transform[subject_index], reading_data_after_trim[subject_index], calibration_data, subject_index)
