import pandas as pd

import ReadData
import configs


def check_split_given_word(word):
    if word.strip() == "" or word in configs.punctuation_list:
        return True
    else:
        return False


def split_text_with_punctuation():
    text_data = ReadData.read_text_data("text_sorted_mapping.csv")
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

            sentence_list = []
            unique_sentence_list = []
            word_index_in_sentence = []
            word_index = 0
            sentence_start_index = 0
            while word_index < row_df.shape[0]:
                if (not check_split_given_word(row_df.iloc[word_index]["word"])) and word_index < row_df.shape[0] - 1:
                    word_index += 1
                else:
                    # if sentence_start_index == word_index and word_index < row_df.shape[0] - 1:
                    if sentence_start_index == word_index and check_split_given_word(row_df.iloc[word_index]["word"]):
                        sentence_start_index += 1
                        word_index += 1
                        sentence_list.append("/split")
                        word_index_in_sentence.append(-1)
                    else:
                        start_index = sentence_start_index
                        if sentence_start_index > 0:
                            start_index -= 1
                        end_index = word_index
                        last_word = row_df.iloc[end_index]["word"]
                        word_list = row_df.iloc[start_index: end_index + 1]["word"].tolist()

                        end_col = row_df.iloc[word_index]["col"]
                        if end_col == row_df.shape[0] - 1:
                            word_list.append("[SEP]")

                        if len(unique_sentence_list) == 0:
                            sentence = "[CLS]" + "".join(word_list)
                        else:
                            sentence = "".join(word_list)

                        if word_index == row_df.shape[0] - 1 and not check_split_given_word(last_word):
                            end_index += 1
                        for index in range(sentence_start_index, end_index):
                            sentence_list.append(sentence)
                            word_index_in_sentence.append(index - sentence_start_index)
                        if word_index == row_df.shape[0] - 1 and check_split_given_word(last_word):
                            sentence_list.append("/split")
                            word_index_in_sentence.append(-1)
                        unique_sentence_list.append(sentence)
                        if word_index == row_df.shape[0] - 1:
                            break
                        sentence_start_index = word_index

            row_df["sentence"] = sentence_list
            row_df["word_index_in_sentence"] = word_index_in_sentence
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