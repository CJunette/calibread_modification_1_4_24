import ast
import multiprocessing
import os.path
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import openai
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
import ReadData
import configs
import torch
from torch import nn


def start_using_IDE():
    """
    When start python file using IDE, we just need to set proxy with openai.
    :return:
    """
    openai.proxy = 'http://127.0.0.1:10809'


def set_openai():
    # openai.organization = "org-veTDIexYdGbOKcYt8GW4SNOH"
    key_file = "keys/openai.txt"
    with open(key_file, "r") as f:
        key = f.read().strip()
        openai.api_key = key


def get_embedding_for_unique_sentence():
    start_using_IDE()
    set_openai()

    unique_sentence_data = ReadData.read_text_data("unique_sentence.csv")
    unique_sentence_data_with_embedding = []

    for text_index in range(len(unique_sentence_data)):
        print(f"processing text {text_index}")
        unique_sentence_df = unique_sentence_data[text_index]
        embedding_list = []
        for sentence_index in range(unique_sentence_df.shape[0]):
            sentence = unique_sentence_df.iloc[sentence_index]["sentence"]
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=sentence,
                encoding_format="float",
            )
            embedding = response.data[0].embedding
            embedding_list.append(embedding)
            time.sleep(1) # 防止过于频繁的请求导致server overload。
        unique_sentence_df["embedding"] = embedding_list
        unique_sentence_data_with_embedding.append(unique_sentence_df)

    return unique_sentence_data_with_embedding


def save_embedding_for_unique_sentence():
    start_using_IDE()
    set_openai()

    save_path_prefix = f"text/{configs.round_num}/embeddings_for_unique_sentence"
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    unique_sentence_data = ReadData.read_text_data("unique_sentence.csv")
    unique_sentence_data_with_embedding = []

    for text_index in range(19, len(unique_sentence_data)):
        print(f"processing text {text_index}")
        unique_sentence_df = unique_sentence_data[text_index]
        embedding_list = []
        for sentence_index in range(unique_sentence_df.shape[0]):
            print(f"processing sentence {sentence_index}/{unique_sentence_df.shape[0]}")
            sentence = unique_sentence_df.iloc[sentence_index]["sentence"]
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=sentence,
                encoding_format="float",
            )
            embedding = response.data[0].embedding
            embedding_list.append(embedding)
            time.sleep(1)  # 防止过于频繁的请求导致server overload。
        unique_sentence_df["embedding"] = embedding_list
        file_name = f"{save_path_prefix}/{text_index}.csv"
        unique_sentence_df.to_csv(file_name, index=False, encoding="utf-8-sig")
        # unique_sentence_data_with_embedding.append(unique_sentence_df)
        time.sleep(1)

    # unique_sentence_data_with_embedding = get_embedding_for_unique_sentence()
    # df = pd.concat(unique_sentence_data_with_embedding, ignore_index=True).reset_index(drop=True)
    # file_path = f"text/{configs.round_num}/unique_sentence_with_embedding.csv"
    # df.to_csv(file_path)
    file_path_list = os.listdir(save_path_prefix)
    file_path_list.sort(key=lambda x: int(x.split(".")[0]))
    unique_sentence_data_with_embedding = []
    for file_path_index in range(len(file_path_list)):
        file_path = f"{save_path_prefix}/{file_path_list[file_path_index]}"
        df = pd.read_csv(file_path)
        unique_sentence_data_with_embedding.append(df)
    df = pd.concat(unique_sentence_data_with_embedding, ignore_index=True).reset_index(drop=True)
    file_path = f"text/{configs.round_num}/unique_sentence_with_embedding.csv"
    df.to_csv(file_path)


def save_embedding_for_tokens(mode="fine"):
    start_using_IDE()
    set_openai()

    token_data = ReadData.read_tokens()

    save_file_path = f"text/{configs.round_num}/embeddings_for_{mode}_tokens"
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    for text_index in range(0, len(token_data)):
        print(f"processing text {text_index}")
        token_df = token_data[text_index]
        embedding_list = []
        for token_index in range(token_df.shape[0]):
            print(f"processing token {token_index}")
            token = token_df.iloc[token_index]["tokens"]
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=token,
                encoding_format="float",
            )
            embedding = response.data[0].embedding
            embedding_list.append(embedding)
            time.sleep(0.5)
        df = token_df[["tokens", "text_unit_component", "row", "split"]].copy()
        df["embedding"] = embedding_list
        # change the column name "text_unit_component" to "col".
        df.rename(columns={"text_unit_component": "col"}, inplace=True)
        df["col"] = df["col"].apply(eval).apply(lambda x: x[0])
        df["row"] = df["row"].apply(eval).apply(lambda x: x[0])

        file_name = f"{save_file_path}/{text_index}.csv"
        df.to_csv(file_name, index=False, encoding="utf-8-sig")

        time.sleep(1)


def get_location_embedding_vector_single_pool(text_data, embedding_data, text_index, row_index, col_index):
    # print(f"processing text {text_index}, row {row_index}, col {col_index}")
    embedding_data_df = embedding_data[text_index]
    embedding_data_df = embedding_data_df[embedding_data_df["row"] == row_index]

    text_df = text_data[text_index]
    row_df = text_df[text_df["row"] == row_index]

    sentence = row_df[row_df["col"] == col_index]["sentence"].tolist()[0]
    if sentence == "/split":
        return None
    embedding = embedding_data_df[embedding_data_df["sentence"] == sentence]["embedding"].tolist()[0]
    embedding = ast.literal_eval(embedding)
    word_index_in_sentence = row_df[row_df["col"] == col_index]["word_index_in_sentence"].tolist()[0]
    vector = [word_index_in_sentence] + embedding

    return text_index, row_index, col_index, np.array(vector)


def get_density_single_pool(density_data, subject_index, text_index, row_index, col_index):
    # print(f"processing subject {subject_index}, text {text_index}, row {row_index}, col {col_index}")
    density_df = density_data[subject_index][text_index]
    density = density_df[(density_df["row"] == row_index) & (density_df["col"] == col_index)]["text_density"].tolist()[0]
    relative_density = density_df[(density_df["row"] == row_index) & (density_df["col"] == col_index)]["relative_text_density"].tolist()[0]

    return subject_index, text_index, row_index, col_index, density, relative_density


def get_density_and_embedding_location_vector(bool_token_embedding=True):
    sentence_embedding_data = ReadData.read_text_data("unique_sentence_with_embedding.csv")
    text_data = ReadData.read_text_data("text_sorted_mapping_with_split.csv")
    print()

    token_embedding_data = ReadData.read_token_embedding()

    # 获取每个word对应的density。
    density_data = ReadData.read_density()

    for text_index in range(len(text_data)):
        text_df = text_data[text_index]
        sentence_embedding_df = sentence_embedding_data[text_index]
        text_df["sentence_embedding"] = None
        if bool_token_embedding:
            text_df["token_embedding"] = None
            text_df["col_within_token"] = -1

        for text_unit_index in range(sentence_embedding_df.shape[0]):
            sentence = sentence_embedding_df.iloc[text_unit_index]["sentence"]
            embedding = sentence_embedding_df.iloc[text_unit_index]["embedding"]
            embedding = ast.literal_eval(embedding)
            for index, series in text_df.loc[text_df["sentence"] == sentence].iterrows():
                text_df.at[index, "sentence_embedding"] = embedding
        if bool_token_embedding:
            for token_index in range(token_embedding_data[text_index].shape[0]):
                token = token_embedding_data[text_index].iloc[token_index]["tokens"]
                # 确认当前的token都是有意义的内容，在将其embedding加入到text_df中。
                bool_break = True
                for token_unit in token:
                    if not (token_unit.strip() == "" or token_unit in configs.punctuation_list):
                        bool_break = False
                        break
                if bool_break:
                    continue

                embedding = token_embedding_data[text_index].iloc[token_index]["embedding"]
                embedding = ast.literal_eval(embedding)
                row = int(token_embedding_data[text_index].iloc[token_index]["row"])
                col = token_embedding_data[text_index].iloc[token_index]["col"]
                col = ast.literal_eval(col)
                for col_index in range(len(col)):
                    index = text_df.loc[(text_df["row"] == row) & (text_df["col"] == col[col_index]), "token_embedding"].index.tolist()[0]
                    text_df.at[index, "token_embedding"] = embedding
                    text_df.at[index, "col_within_token"] = col_index

    vector_list = []
    for text_index in range(len(text_data)):
        text_df = text_data[text_index]
        text_df = text_df[text_df["sentence"] != "/split"]
        row_list = text_df["row"].tolist()
        col_list = text_df["col"].tolist()
        word_index_in_sentence_list = text_df["word_index_in_sentence"].tolist()
        sentence_embedding_list = text_df["sentence_embedding"].tolist()
        sentence_embedding_list = [np.array([word_index_in_sentence_list[i]] + sentence_embedding_list[i]) for i in range(len(sentence_embedding_list))]
        text_index_list = [text_index for _ in range(len(row_list))]
        if bool_token_embedding:
            token_embedding_list = text_df["token_embedding"].tolist()
            col_within_token_list = text_df["col_within_token"].tolist()
            token_embedding_list = [np.array([col_list[i]] + [col_within_token_list[i]] + token_embedding_list[i]) for i in range(len(token_embedding_list))]
            zip_list = list(zip(text_index_list, row_list, col_list, sentence_embedding_list, token_embedding_list))
        else:
            zip_list = list(zip(text_index_list, row_list, col_list, sentence_embedding_list))
        zip_list = np.array(zip_list)
        vector_list.extend(zip_list)
    vector_list = np.array(vector_list)

    density_list = []
    for subject_index in range(len(density_data)):
        density_list_1 = []
        for text_index in range(len(density_data[subject_index])):
            density_df = density_data[subject_index][text_index].copy()
            density_df = density_df[density_df["word"] != "blank_supplement"]
            text_df = text_data[text_index]
            text_df_first_index = text_df.index.tolist()[0]
            text_df = text_df[text_df["sentence"] != "/split"]
            text_df_indices = text_df.index.tolist()
            text_df_indices = [index - text_df_first_index for index in text_df_indices]
            density_df = density_df.iloc[text_df_indices]

            row_list = density_df["row"].tolist()
            col_list = density_df["col"].tolist()
            density = density_df["text_density"].tolist()
            relative_density = density_df["relative_text_density"].tolist()
            text_index_list = [text_index for _ in range(len(row_list))]
            subject_index_list = [subject_index for _ in range(len(row_list))]
            zip_list = list(zip(subject_index_list, text_index_list, row_list, col_list, density, relative_density))
            zip_list = np.array(zip_list)
            density_list_1.extend(zip_list)
        density_list.append(np.array(density_list_1))
    density_list = np.array(density_list)

    # # 获取每个word的word_index_in_sentence和embedding组成的vector。
    # print("embedding_vector_list start")
    # # vector_list = []
    # embedding_args_list = []
    #
    # # for text_index in text_index_list:
    # for text_index in range(len(text_data)):
    #     text_df = text_data[text_index]
    #     row_list = text_df["row"].unique().tolist()
    #     row_list.sort()
    #     for row_index in row_list:
    #         # embedding_data_df = sentence_embedding_data[text_index]
    #         # embedding_data_df = embedding_data_df[embedding_data_df["row"] == row_index]
    #
    #         row_df = text_df[text_df["row"] == row_index]
    #         col_list = row_df["col"].unique().tolist()
    #         col_list.sort()
    #         for col_index in col_list:
    #             sentence = row_df[row_df["col"] == col_index]["sentence"].tolist()[0]
    #             if sentence == "/split":
    #                 continue
    #             # embedding = embedding_data_df[embedding_data_df["sentence"] == sentence]["embedding"].tolist()[0]
    #             # embedding = ast.literal_eval(embedding)
    #             # word_index_in_sentence = row_df[row_df["col"] == col_index]["word_index_in_sentence"].tolist()[0]
    #             # vector = [word_index_in_sentence] + embedding
    #             # vector_list.append(vector_list)
    #
    #             embedding_args_list.append((text_data, sentence_embedding_data, text_index, row_index, col_index))
    #
    # with multiprocessing.Pool(processes=configs.number_of_process) as pool:
    #     vector_list = pool.starmap(get_location_embedding_vector_single_pool, embedding_args_list)
    # vector_list = [vector for vector in vector_list if vector[0] is not None]
    # vector_list = np.array(vector_list)
    # print("embedding_vector_list finished")
    #
    # print("density_list start")
    # density_args_list = []
    # for subject_index in range(len(density_data)):
    #     for vector_index in range(len(vector_list)):
    #         text_index = vector_list[vector_index][0]
    #         row_index = vector_list[vector_index][1]
    #         col_index = vector_list[vector_index][2]
    #         # density_df = density_data[subject_index][text_index]
    #         # density = density_df[(density_df["row"] == row_index) & (density_df["col"] == col_index)]["density"].tolist()[0]
    #         density_args_list.append((density_data, subject_index, text_index, row_index, col_index))
    #
    # with multiprocessing.Pool(processes=configs.number_of_process) as pool:
    #     density_list = pool.starmap(get_density_single_pool, density_args_list)
    # density_list = np.array(density_list)
    # density_list = density_list.reshape((len(density_data), len(vector_list), -1))
    # print("density_list finished")

    return vector_list, density_list, sentence_embedding_data, text_data, density_data


def cluster_word_given_embedding():
    text_index_list = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15]
    vector_list, density_list, embedding_data, text_data, density_data = get_density_and_embedding_location_vector(False)

    print("tsne start")
    vector_only = np.vstack(vector_list[:, 3])
    # 不考虑每个subject和word的density。
    tsne = TSNE(n_components=2, perplexity=30, random_state=configs.random_seed, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(vector_only)
    print("tsne finished")

    # 可视化。
    word_index_in_sentence_list = vector_only[:, 0].astype(np.int32)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # 红、绿、蓝
    n_bins = 5000  # 分段数
    cmap_name = "custom_gradient"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    color_list = []
    for subject_index in range(len(density_list)):
        density = density_list[subject_index][:, 4]
        relative_density = density_list[subject_index][:, 5]
        relative_density_ceil = np.percentile(relative_density, 80)
        relative_density = relative_density / relative_density_ceil
        color_list.append([cm(x) for x in relative_density])

    fig = plt.figure(figsize=(24, 16))
    ax = fig.add_subplot(111)

    for subject_index in range(len(density_list)):
        ax.scatter(vis_dims[:, 0], vis_dims[:, 1], c=color_list[subject_index], s=10, alpha=0.2)

    # 考虑每个subject和word的density。
    # tsne = TSNE(n_components=2, perplexity=90, random_state=configs.random_seed, init='random', learning_rate=3)
    # vector_with_density = []
    # for subject_index in range(len(density_list)):
    #     vector_with_density.append(np.concatenate([vector_only, density_list[subject_index][:, 5].reshape(-1, 1)], axis=1))
    # vector_with_density = np.concatenate(vector_with_density, axis=0)
    #
    # vis_dims = tsne.fit_transform(vector_with_density)
    # print("tsne finished")
    #
    # # 可视化。
    # word_index_in_sentence_list = vector_only[:, 0].astype(np.int32)
    # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # 红、绿、蓝
    # n_bins = 100  # 分段数
    # cmap_name = "custom_gradient"
    # cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    # gradient_colors = cm(np.linspace(0, 1, 30)) # 提取30个颜色
    # gradient_colors = gradient_colors[:, :3]
    # color_list = [gradient_colors[word_index_in_sentence] for word_index_in_sentence in word_index_in_sentence_list] * len(density_list)
    # fig = plt.figure(figsize=(16, 12))
    # ax = fig.add_subplot(111)
    #
    # alpha_list = density_list[:, :, 5].reshape(-1)
    # # alpha_list = alpha_list / np.max(alpha_list) / len(density_list)
    # alpha_list = alpha_list / np.max(alpha_list) / 5
    # ax.scatter(vis_dims[:, 0], vis_dims[:, 1], c=color_list, s=10, alpha=alpha_list)

    # plt.show()
    file_prefix = f"pic/tsne/{configs.round_num}"
    if not os.path.exists(file_prefix):
        os.makedirs(file_prefix)
    plt.savefig(f"{file_prefix}/test_003.png")
    plt.clf()
    plt.close()


def linear_regression():
    '''
    效果不好，R^2只有0.1多一些。
    :return:
    '''
    text_index_list = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15]
    vector_list, density_list, embedding_data, text_data, density_data = get_density_and_embedding_location_vector(False)

    X = [vector_list[i][3] for i in range(len(vector_list)) if vector_list[i][0] in text_index_list]
    X = X * len(density_data)
    X = np.array(X)

    y = []
    density_index_list = []
    for density_index in range(len(density_list[0])):
        if int(density_list[0][density_index][1]) in text_index_list:
            density_index_list.append(density_index)
    for subject_index in range(len(density_list)):
        density_list_1 = density_list[subject_index][density_index_list][:, 4]
        # density_list_1 = []
        # for density_index in density_index_list:
        #     density_list_1.append(density_list[subject_index][density_index][4])
        y.extend(density_list_1)

    model = LinearRegression()
    model.fit(X, y)
    print("Coefficients: ", model.coef_)
    print("Intercept: ", model.intercept_)
    print("Score: ", model.score(X, y))


class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_layers_1=64, hidden_layers_2=128, hidden_layer_3=256, hidden_layer_4=512, hidden_layer_5=1024):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers_1)
        self.bn1 = nn.BatchNorm1d(hidden_layers_1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers_1, hidden_layers_2)
        self.bn2 = nn.BatchNorm1d(hidden_layers_2)
        self.fc3 = nn.Linear(hidden_layers_2, hidden_layer_3)
        self.bn3 = nn.BatchNorm1d(hidden_layer_3)
        self.fc4 = nn.Linear(hidden_layer_3, hidden_layer_4)
        self.bn4 = nn.BatchNorm1d(hidden_layer_4)
        self.fc5 = nn.Linear(hidden_layer_4, 1)
        # self.bn5 = nn.BatchNorm1d(hidden_layer_5)
        # self.fc6 = nn.Linear(hidden_layer_5, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.fc5(out)
        # out = self.bn5(out)
        # out = self.relu(out)
        # out = self.fc6(out)
        return out


def train(model, X, y, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


def select_data_given_indices(index_list, vector_list, density_list, bool_token_embedding=True):
    if bool_token_embedding:
        X = [np.concatenate([vector_list[i][3], vector_list[i][4]]) for i in range(len(vector_list)) if vector_list[i][0] in index_list]
    else:
        X = [vector_list[i][3] for i in range(len(vector_list)) if vector_list[i][0] in index_list]
    X = X * len(density_list)
    X = np.array(X)
    X_info = [vector_list[i][:3] for i in range(len(vector_list)) if vector_list[i][0] in index_list]
    X_info = X_info * len(density_list)
    X_info = np.array(X_info)

    y = []
    density_index_list = []
    for density_index in range(len(density_list[0])):
        if int(density_list[0][density_index][1]) in index_list:
            density_index_list.append(density_index)
    for subject_index in range(len(density_list)):
        density_list_1 = density_list[subject_index][density_index_list][:, 4]
        y.extend(density_list_1)

    return X, y, X_info


def linear_neural_network(bool_token_embedding=True):
    '''
    :return:
    '''
    torch.manual_seed(configs.random_seed)

    training_index_list = configs.training_index_list
    vector_list, density_list, embedding_data, text_data, density_data = get_density_and_embedding_location_vector(bool_token_embedding)

    X_train, y_train, X_train_info = select_data_given_indices(training_index_list, vector_list, density_list, bool_token_embedding)
    # 转换数据为PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to('cuda')
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to('cuda')

    # 定义模型
    input_size = X_train.shape[1]
    if bool_token_embedding:
        model = SimpleNet(input_size, hidden_layers_1=64, hidden_layers_2=128, hidden_layer_3=256, hidden_layer_4=512).to('cuda')
    else:
        model = SimpleNet(input_size, hidden_layers_1=64, hidden_layers_2=128, hidden_layer_3=256, hidden_layer_4=512).to('cuda')

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 训练模型
    train(model, X_train_tensor, y_train_tensor, optimizer, criterion, epochs=300)

    validation_index_list = np.setdiff1d(np.array([i for i in range(40)]), np.array(training_index_list))
    X_val, y_val, X_val_info = select_data_given_indices(validation_index_list, vector_list, density_list, bool_token_embedding)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to('cuda')
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to('cuda')
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_tensor)
        loss = criterion(y_val_pred, y_val_tensor)
        print(f'Validation loss: {loss.item()}')

    return model, training_index_list, validation_index_list, X_train, y_train, X_val, y_val, X_train_info, X_val_info, vector_list, density_list, embedding_data, text_data, density_data


def visualize_linear_neural_network_prediction_using_sentence_unit(vector_list, density_list, text_data, model_result, training_index_list, save_index_str="000"):
    word_index = 0
    start_word_index = 0
    picture_index = 0
    while word_index < len(vector_list):
        print(f"processing word {word_index}/{len(vector_list)}")

        text_index = vector_list[word_index][0]
        row_index = vector_list[word_index][1]
        start_col_index = vector_list[word_index][2]

        if text_index in training_index_list:
            prefix = "training"
        else:
            prefix = "validation"

        end_col_index = start_col_index
        probe_index = word_index + 1
        while probe_index < len(vector_list):
            vector = vector_list[probe_index][3]
            word_index_in_sentence = vector[0]
            if word_index_in_sentence == 0:
                end_row_index = vector_list[probe_index][1]
                if end_row_index != row_index:  # 如果结束时换行了，新的对象的col_index就会是0，这样会有问题。所以这里需要区分一下。
                    end_col_index += 1
                else:
                    end_col_index = vector_list[probe_index][2]
                break
            else:
                end_col_index = vector_list[probe_index][2]
                probe_index += 1

        if probe_index == len(vector_list):
            end_col_index += 1

        text_df = text_data[text_index]
        row_df = text_df[text_df["row"] == row_index]

        if start_col_index > 0:
            word_list = row_df[(row_df["col"] >= start_col_index - 1) & (row_df["col"] < end_col_index)]["word"].tolist()
        else:
            word_list = row_df[(row_df["col"] >= start_col_index) & (row_df["col"] < end_col_index)]["word"].tolist()

        density_list_1 = []
        for subject_index in range(len(density_list)):
            density_list_2 = []
            if start_col_index > 0:  # 如果start_col_index不为0，则说明前面应该有一个\split的内容，将其密度-1添加进来。
                density_list_2.append(-1)
            for index in range(start_word_index, probe_index):
                density_list_2.append(density_list[subject_index][index][4])
            if probe_index == len(vector_list) or vector_list[probe_index][1] == row_index:  # 如果结束的位置是同行，代表末尾还会有若干个\split，所以末尾也要加上这些-1。
                length = len(word_list) - len(density_list_2)
                for index in range(length):
                    density_list_2.append(-1)

            density_list_1.append(np.array(density_list_2))
        density_list_1 = np.array(density_list_1)

        prediction_list = []
        if start_col_index > 0:  # 如果start_col_index不为0，则说明前面应该有一个\split的内容，将其密度-1添加进来。
            prediction_list.append(-1)
        for col_index in range(start_col_index, end_col_index):
            prediction_list.append(model_result[text_index][row_index][col_index])

        word_index = probe_index
        start_word_index = word_index

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        index_list = [i for i in range(len(word_list))]
        ax.scatter(index_list, prediction_list, c='r', marker='o', s=10, label="prediction")
        ax.plot(index_list, prediction_list, c='r')
        for subject_index in range(len(density_list_1)):
            ax.scatter(index_list, density_list_1[subject_index], c='g', marker='x', s=10, label="ground truth")
        density_list_1_mean = np.mean(density_list_1, axis=0)
        ax.plot(index_list, density_list_1_mean, c='g')

        save_path = f"pic/simple_linear_prediction/sentence_unit/{save_index_str}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = f"{save_path}/{prefix}-text_index_{text_index}-picture_index_{picture_index}.png"
        picture_index += 1
        plt.legend()
        ax.set_xticks(index_list)
        ax.set_xticklabels(word_list)
        plt.title(f"{prefix} text_index: {text_index}")
        # plt.show()
        plt.savefig(save_name)
        plt.clf()
        plt.close()


def visualize_linear_neural_network_prediction_using_row_unit(vector_list, density_data, text_data, model_result, training_index_list, save_index_str="000"):
    word_index = 0
    start_word_index = 0
    picture_index = 0
    while word_index < len(vector_list):
        print(f"processing word {word_index}/{len(vector_list)}")

        text_index = vector_list[word_index][0]
        row_index = vector_list[word_index][1]

        if text_index in training_index_list:
            prefix = "training"
        else:
            prefix = "validation"

        probe_index = word_index + 1
        while probe_index < len(vector_list):
            end_row_index = vector_list[probe_index][1]
            if end_row_index != row_index:
                break
            probe_index += 1

        text_df = text_data[text_index]
        row_df = text_df[text_df["row"] == row_index].copy()
        row_df["density"] = -1
        row_df["prediction"] = -1
        word_list = row_df["word"].tolist()

        prediction_list = []
        start_col = row_df["col"].tolist()[0]
        end_col = row_df["col"].tolist()[-1]
        for col_index in range(start_col, end_col + 1):
            prediction_list.append(model_result[text_index][row_index][col_index])

        density_list_1 = []
        for subject_index in range(len(density_data)):
            density_df = density_data[subject_index][text_index]
            density_df = density_df[density_df["word"] != "blank_supplement"]
            density_df = density_df[density_df["row"] == row_index]
            # for index in range(start_word_index, probe_index):
            #     density_list_2.append(density_list[subject_index][index][4])
            # row_df.loc[row_df["sentence"] != "/split", "density"] = density_list_2
            # density_list_2 = row_df["density"].tolist()
            density_list_2 = density_df["text_density"].tolist()
            density_list_1.append(np.array(density_list_2))
        density_list_1 = np.array(density_list_1)

        word_index = probe_index
        start_word_index = word_index

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        index_list = [i for i in range(len(word_list))]
        ax.scatter(index_list, prediction_list, c='r', marker='o', s=10, label="prediction")
        ax.plot(index_list, prediction_list, c='r')
        for subject_index in range(len(density_list_1)):
            ax.scatter(index_list, density_list_1[subject_index], c='g', marker='x', s=10, label="ground truth")
        density_list_1_mean = np.mean(density_list_1, axis=0)
        ax.plot(index_list, density_list_1_mean, c='g')

        save_path = f"pic/simple_linear_prediction/row_unit/{save_index_str}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = f"{save_path}/{prefix}-text_index_{text_index}-row_index_{row_index}-picture_index_{picture_index}.png"
        picture_index += 1
        plt.legend()
        ax.set_xticks(index_list)
        ax.set_xticklabels(word_list)
        plt.title(f"{prefix} text_index: {text_index}")
        # plt.show()
        plt.savefig(save_name)
        plt.clf()
        plt.close()


def visualize_linear_neural_network_prediction():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    (model, training_index_list, validation_index_list,
     X_train, y_train, X_val, y_val, X_train_info, X_val_info,
     vector_list, density_list, embedding_data, text_data, density_data) = linear_neural_network(bool_token_embedding=True)

    # get y_train_pred and y_val_pred.
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to('cuda')
        y_train_pred = model(X_train_tensor).to('cpu')
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to('cuda')
        y_val_pred = model(X_val_tensor).to('cpu')

    model_result = [[[-1 for _ in range(configs.col_num)] for _ in range(configs.row_num)] for _ in range(len(text_data))]
    for index in range(len(X_train_info)):
        text_index = X_train_info[index][0]
        row_index = X_train_info[index][1]
        col_index = X_train_info[index][2]
        model_result[text_index][row_index][col_index] = y_train_pred[index]
    for index in range(len(X_val_info)):
        text_index = X_val_info[index][0]
        row_index = X_val_info[index][1]
        col_index = X_val_info[index][2]
        model_result[text_index][row_index][col_index] = y_val_pred[index]

    # visualize_linear_neural_network_prediction_using_sentence_unit(vector_list, density_list, text_data, model_result, training_index_list)
    visualize_linear_neural_network_prediction_using_row_unit(vector_list, density_data, text_data, model_result, training_index_list, save_index_str="004")



