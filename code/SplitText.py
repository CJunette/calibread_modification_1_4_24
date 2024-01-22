import pandas as pd

import ReadData
import configs


def check_split_given_word(word):
    if word.strip() == "" or word in configs.punctuation_list:
        return True
    else:
        return False


def add_word(start_index, end_index, row_df, sentence_list, word_index_in_sentence, unique_sentence_list, sentence_length_list):
    clip_start_index = max(0, start_index - 1)
    clip_end_index = min(row_df.shape[0], end_index + 1)

    word_list = row_df.iloc[clip_start_index: clip_end_index]["word"].tolist()
    if len(word_list) == 0:
        return sentence_list, word_index_in_sentence, unique_sentence_list

    if clip_start_index == 0:
        sentence = "[CLS]" + "".join(word_list)
    else:
        sentence = "".join(word_list)
    if clip_end_index == row_df.shape[0]:
        sentence += "[SEP]"

    for index in range(start_index, end_index):
        sentence_list.append(sentence)
        word_index_in_sentence.append(index - start_index)
        sentence_length_list.append(end_index - start_index)
    unique_sentence_list.append(sentence)

    return sentence_list, word_index_in_sentence, unique_sentence_list, sentence_length_list


def find_interval(num, interval_list):
    for index in range(0, len(interval_list)):
        min_in_interval = interval_list[index][0]
        max_in_interval = interval_list[index][-1]
        if min_in_interval <= num <= max_in_interval:
            return index
    return -1


def add_split(start_index, end_index, row_df, token_row_df, sentence_list, word_index_in_sentence, unique_sentence_list, sentence_length_list):
    token_col_list = token_row_df["col"].tolist()
    start = start_index
    end = end_index - 1

    interval_start = find_interval(start, token_col_list)
    interval_end = find_interval(end, token_col_list)

    word_list = row_df.iloc[start_index: end_index]["word"].tolist()
    if interval_start == 0:
        sentence = "[CLS]" + "".join(word_list)
    else:
        sentence = token_row_df.iloc[interval_start - 1]["tokens"] + "".join(word_list)
    if interval_end == len(token_col_list) - 1:
        sentence += "[SEP]"
    else:
        sentence += token_row_df.iloc[interval_end + 1]["tokens"]

    for index in range(start_index, end_index):
        sentence_list.append(sentence)
        # word_index_in_sentence.append(index - start_index)
        word_index_in_sentence.append(-1 * (index - start_index + 1))
        sentence_length_list.append(end_index - start_index)
    unique_sentence_list.append(sentence)

    return sentence_list, word_index_in_sentence, unique_sentence_list, sentence_length_list


def get_start_and_end_index_of_split(split_indices, split_index):
    '''
    该函数的目的是找到split_indices中，从split_index开始，连续的split的起始和终止位置。
    起始位置start_index就是正常的split_indices[split_index]。
    终止位置end_index需要用probe_index去向前探索，直到找到不连续的split。
    :param split_indices:
    :param split_index:
    :return:
    '''
    start_index = split_indices[split_index]
    probe_index = split_index + 1
    while probe_index < len(split_indices) and split_indices[probe_index] - split_indices[probe_index - 1] == 1:
        probe_index += 1
    probe_index -= 1
    end_index = split_indices[probe_index] + 1

    return start_index, end_index, probe_index


def split_text_with_punctuation():
    text_data = ReadData.read_text_data("text_sorted_mapping.csv")
    token_data = ReadData.read_tokens("fine")
    unique_sentence_df_list_1 = []
    for text_index in range(0, len(text_data)):
        text_df = text_data[text_index]
        rows = text_df["row"].unique().tolist()
        rows.sort()
        new_row_df_list = []
        unique_sentence_df_list_2 = []
        for row_index in range(0, len(rows)):
            row = rows[row_index]
            row_df = text_df[text_df["row"] == row].reset_index(drop=True)
            row_df_split = row_df[row_df["word"].apply(check_split_given_word)]
            split_indices = row_df_split.index.tolist()

            sentence_list = []
            unique_sentence_list = []
            word_index_in_sentence = []
            sentence_length = []

            # 如果len(split_indices) == 0，代表根本没有标点，此时可以直接把整个row_df作为一个sentence。
            if len(split_indices) == 0:
                word_list = row_df["word"].tolist()
                word_str = "[CLS]" + "".join(word_list) + "[SEP]"
                for index in range(0, row_df.shape[0]):
                    sentence_list.append(word_str)
                    word_index_in_sentence.append(index)
                    unique_sentence_list.append(word_str)
                    sentence_length.append(row_df.shape[0])

            token_row_df = token_data[text_index][token_data[text_index]["row"] == row].reset_index(drop=True)

            split_index = 0
            while split_index < len(split_indices):
                # 以split_indices为[1, 2, 6, 14, 21]为例。
                # split_index是用于确定split_indices中的位置的。
                # start_index和end_index是用于确定row_df中的位置的。
                if split_index == 0 and split_indices[split_index] != 0:
                    # 先添加0到第一个split之间的内容。即[0, 1]。
                    start_index = 0
                    end_index = split_indices[split_index]
                    sentence_list, word_index_in_sentence, unique_sentence_list, sentence_length = add_word(start_index, end_index, row_df, sentence_list, word_index_in_sentence, unique_sentence_list, sentence_length)
                    # 然后添加split。即[1, 2]。此时通过get_start_and_end_index_of_split得到的start_index为1，end_index为3。
                    start_index, end_index, probe_index = get_start_and_end_index_of_split(split_indices, split_index)
                    sentence_list, word_index_in_sentence, unique_sentence_list, sentence_length = add_split(start_index, end_index, row_df, token_row_df, sentence_list, word_index_in_sentence, unique_sentence_list, sentence_length)
                    # 接着添加split到下一个split之间的内容。即[2, 6]之间的内容。
                    start_index = end_index
                    if probe_index + 1 < len(split_indices):
                        end_index = split_indices[probe_index + 1]
                    else:
                        end_index = row_df.shape[0]
                    sentence_list, word_index_in_sentence, unique_sentence_list, sentence_length = add_word(start_index, end_index, row_df, sentence_list, word_index_in_sentence, unique_sentence_list, sentence_length)
                # 以split_indices为[0, 1, 6, 14, 21]为例。
                else:
                    # 添加split。即[0, 1]。
                    start_index, end_index, probe_index = get_start_and_end_index_of_split(split_indices, split_index)
                    sentence_list, word_index_in_sentence, unique_sentence_list, sentence_length = add_split(start_index, end_index, row_df, token_row_df, sentence_list, word_index_in_sentence, unique_sentence_list, sentence_length)
                    # 添加split到下一个split之间的内容。即[1, 6]之间的内容。
                    start_index = end_index
                    if probe_index + 1 < len(split_indices):
                        end_index = split_indices[probe_index + 1]
                    else:
                        end_index = row_df.shape[0]
                    sentence_list, word_index_in_sentence, unique_sentence_list, sentence_length = add_word(start_index, end_index, row_df, sentence_list, word_index_in_sentence, unique_sentence_list, sentence_length)

                split_index = probe_index + 1

            # row = rows[row_index]
            # row_df = text_df[text_df["row"] == row].reset_index(drop=True)
            #
            # sentence_list = []
            # unique_sentence_list = []
            # word_index_in_sentence = []
            # word_index = 0
            # sentence_start_index = 0
            # while word_index < row_df.shape[0]:
            #     if (not check_split_given_word(row_df.iloc[word_index]["word"])) and word_index < row_df.shape[0] - 1:
            #         word_index += 1
            #     else:
            #         # if sentence_start_index == word_index and word_index < row_df.shape[0] - 1:
            #         if sentence_start_index == word_index and check_split_given_word(row_df.iloc[word_index]["word"]):
            #             sentence_start_index += 1
            #             word_index += 1
            #             sentence_list.append("/split")
            #             word_index_in_sentence.append(-1)
            #         else:
            #             start_index = sentence_start_index
            #             if sentence_start_index > 0:
            #                 start_index -= 1
            #             end_index = word_index
            #             last_word = row_df.iloc[end_index]["word"]
            #             word_list = row_df.iloc[start_index: end_index + 1]["word"].tolist()
            #
            #             end_col = row_df.iloc[word_index]["col"]
            #             if end_col == row_df.shape[0] - 1:
            #                 word_list.append("[SEP]")
            #
            #             if len(unique_sentence_list) == 0:
            #                 sentence = "[CLS]" + "".join(word_list)
            #             else:
            #                 sentence = "".join(word_list)
            #
            #             if word_index == row_df.shape[0] - 1 and not check_split_given_word(last_word):
            #                 end_index += 1
            #             for index in range(sentence_start_index, end_index):
            #                 sentence_list.append(sentence)
            #                 word_index_in_sentence.append(index - sentence_start_index)
            #             if word_index == row_df.shape[0] - 1 and check_split_given_word(last_word):
            #                 sentence_list.append("/split")
            #                 word_index_in_sentence.append(-1)
            #             unique_sentence_list.append(sentence)
            #             if word_index == row_df.shape[0] - 1:
            #                 break
            #             sentence_start_index = word_index

            row_df["sentence"] = sentence_list
            row_df["word_index_in_sentence"] = word_index_in_sentence
            row_df["sentence_length"] = sentence_length
            new_row_df_list.append(row_df)

            para_id_list = [text_index] * len(unique_sentence_list)
            row_id_list = [row_index] * len(unique_sentence_list)
            unique_sentence_df = pd.DataFrame({"para_id": para_id_list, "row": row_id_list, "sentence": unique_sentence_list})
            unique_sentence_df_list_2.append(unique_sentence_df)

        new_text_df = pd.concat(new_row_df_list, ignore_index=True).reset_index(drop=True)
        text_data[text_index] = new_text_df

        unique_sentence_df = pd.concat(unique_sentence_df_list_2, ignore_index=True).reset_index(drop=True)
        unique_sentence_df_list_1.append(unique_sentence_df)

    return text_data, unique_sentence_df_list_1


def save_text_data_after_splitting():
    text_data_after_split, unique_sentence = split_text_with_punctuation()
    file_path_prefix = f"text/{configs.round_num}"

    text_df = pd.concat(text_data_after_split, ignore_index=True).reset_index(drop=True)
    text_df.to_csv(f"{file_path_prefix}/text_sorted_mapping_with_split.csv", index=False, encoding="utf-8-sig")

    unique_sentence_df = pd.concat(unique_sentence, ignore_index=True).reset_index(drop=True)
    unique_sentence_df.to_csv(f"{file_path_prefix}/unique_sentence.csv", index=False, encoding="utf-8-sig")

