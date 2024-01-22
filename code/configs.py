round_num = "round_5"
random_seed = 0
number_of_process = 8

punctuation_list = {'\'', '\"', '!', '?', '.', '/', '\\', '-', '，', ':', '：', '。', '……', '！', '？', '——', '（', '）', '【', '】', '“', '”', '’', '‘', '：', '；', '《', '》', '、', '—', '～', '·', '「', '」', '『', '』', '…'}

location_penalty = 1
punctuation_penalty = 1
empty_penalty = -1
completion_weight = 10
right_down_corner_unmatched_ratio = 1
left_boundary_ratio = 1
right_boundary_ratio = 10
left_boundary_distance_threshold_ratio = 1.25
right_boundary_distance_threshold_ratio = 1
gradient_descent_stop_accuracy = 0.01

row_num = 6
col_num = 30

left_top_text_center = [380, 272]
right_down_text_center = [1540, 592]
text_width = 40
text_height = 64

training_index_list = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# training_index_list = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

model_density_index = 4 # 这里的4是density，5是relative_density。


