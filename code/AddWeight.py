import configs


def add_weight_to_text(text_data):
    for text_index in range(len(text_data)):
        text_penalty = []
        df = text_data[text_index]
        for index, row in df.iterrows():
            word = row["word"]
            col_index = row["col"]
            row_index = row["row"]
            penalty = 1
            # 添加的点按空白处理。
            if word == "blank_supplement":
                # 对于最左上端的点，惩罚项为1，保证它能够吸引reading points。
                if row_index < 0 and col_index < configs.col_num / 6:
                    penalty = 1
                # 对于其他点，惩罚项为负，保证他们会排斥reading points。
                else:
                    penalty = min(penalty, configs.empty_penalty)
            else:
                # 非添加的点分多种情况处理。惩罚项最终按更大的计算。
                # 1. 标点符号
                if word in configs.punctuation_list:
                    penalty = min(penalty, configs.punctuation_penalty)
                # 2. 空白
                if word.strip() == "":
                    penalty = min(penalty, configs.empty_penalty)
                # 3. 位置靠近行尾。
                # if df[(df["row"] == row_index) & (df["col"] == col_index + 1)]["word"].shape[0] > 0 and df[(df["row"] == row_index) & (df["col"] == col_index + 1)]["word"].tolist()[0] == "blank_supplement":
                #     penalty = min(penalty, configs.location_penalty)
                # 4. 位置靠近行首。
                # if (df[(df["row"] == row_index) & (df["col"] == col_index - 1)]["word"].shape[0] > 0 and
                #         (df[(df["row"] == row_index) & (df["col"] == col_index - 1)]["word"].tolist()[0] == "blank_supplement" or
                #          df[(df["row"] == row_index) & (df["col"] == col_index - 1)]["word"].tolist()[0].strip() == "")):
                #     penalty = min(penalty, configs.location_penalty)
            text_penalty.append(penalty)
        df["penalty"] = text_penalty
        text_data[text_index] = df
    return text_data

