round_num = "round_5"
random_seed = 0
number_of_process = 8
file_index = 34

punctuation_list = {'\'', '\"', '!', '?', '.', '/', '\\', '-', '，', ':', '：', '。', '……', '！', '？', '——', '（', '）', '【', '】', '“', '”', '’', '‘', '：', '；', '《', '》', '、', '—', '～', '·', '「', '」', '『', '』', '…'}

location_penalty = 1
punctuation_penalty = 1
empty_penalty = -0.001
completion_weight = 0.1
right_down_corner_unmatched_ratio = 1
left_boundary_ratio = 125
right_boundary_ratio = 3000
bottom_boundary_ratio = 400
left_boundary_distance_threshold_ratio = 1.25
right_boundary_distance_threshold_ratio = 1
bottom_boundary_distance_threshold_ratio = 1
gradient_descent_stop_accuracy = 0.01
weight_divisor = 5
weight_intercept = 0.05

row_num = 6
col_num = 30

left_top_text_center = [380, 272]
right_down_text_center = [1540, 592]
text_width = 40
text_height = 64

model_density_index = 4 # 这里的4是density，5是relative_density。

bool_weight = True

# training_index_list = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] # 用所有15个训练样本时，剩下的25个样本。
# training_index_list = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# training_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
# 训练1篇文章的3种情况。
training_index_list = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
# training_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
# training_index_list = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
