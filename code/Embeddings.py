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
    unique_sentence_data_with_embedding = get_embedding_for_unique_sentence()
    df = pd.concat(unique_sentence_data_with_embedding, ignore_index=True).reset_index(drop=True)
    file_path = f"text/{configs.round_num}/unique_sentence_with_embedding.csv"
    df.to_csv(file_path)


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


def get_density_and_embedding_location_vector():
    embedding_data = ReadData.read_text_data("unique_sentence_with_embedding.csv")
    text_data = ReadData.read_text_data("text_sorted_mapping_with_split.csv")
    print()

    # 获取每个word对应的density。
    density_data = ReadData.read_density()

    # 获取每个word的word_index_in_sentence和embedding组成的vector。
    print("embedding_vector_list start")
    # vector_list = []
    embedding_args_list = []

    # for text_index in text_index_list:
    for text_index in range(len(text_data)):
        text_df = text_data[text_index]
        row_list = text_df["row"].unique().tolist()
        row_list.sort()
        for row_index in row_list:
            # embedding_data_df = embedding_data[text_index]
            # embedding_data_df = embedding_data_df[embedding_data_df["row"] == row_index]

            row_df = text_df[text_df["row"] == row_index]
            col_list = row_df["col"].unique().tolist()
            col_list.sort()
            for col_index in col_list:
                sentence = row_df[row_df["col"] == col_index]["sentence"].tolist()[0]
                if sentence == "/split":
                    continue
                # embedding = embedding_data_df[embedding_data_df["sentence"] == sentence]["embedding"].tolist()[0]
                # embedding = ast.literal_eval(embedding)
                # word_index_in_sentence = row_df[row_df["col"] == col_index]["word_index_in_sentence"].tolist()[0]
                # vector = [word_index_in_sentence] + embedding
                # vector_list.append(vector_list)

                embedding_args_list.append((text_data, embedding_data, text_index, row_index, col_index))

    with multiprocessing.Pool(processes=configs.number_of_process) as pool:
        vector_list = pool.starmap(get_location_embedding_vector_single_pool, embedding_args_list)
    vector_list = [vector for vector in vector_list if vector[0] is not None]
    vector_list = np.array(vector_list)
    print("embedding_vector_list finished")

    print("density_list start")
    density_args_list = []
    for subject_index in range(len(density_data)):
        for vector_index in range(len(vector_list)):
            text_index = vector_list[vector_index][0]
            row_index = vector_list[vector_index][1]
            col_index = vector_list[vector_index][2]
            # density_df = density_data[subject_index][text_index]
            # density = density_df[(density_df["row"] == row_index) & (density_df["col"] == col_index)]["density"].tolist()[0]
            density_args_list.append((density_data, subject_index, text_index, row_index, col_index))

    with multiprocessing.Pool(processes=configs.number_of_process) as pool:
        density_list = pool.starmap(get_density_single_pool, density_args_list)
    density_list = np.array(density_list)
    density_list = density_list.reshape((len(density_data), len(vector_list), -1))
    print("density_list finished")

    return vector_list, density_list, embedding_data, text_data, density_data


def cluster_word_given_embedding():
    text_index_list = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15]
    vector_list, density_list, embedding_data, text_data, density_data = get_density_and_embedding_location_vector()

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
    vector_list, density_list, embedding_data, text_data, density_data = get_density_and_embedding_location_vector()

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
    def __init__(self, input_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
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


def linear_neural_network():
    '''
    :return:
    '''
    text_index_list = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    vector_list, density_list, embedding_data, text_data, density_data = get_density_and_embedding_location_vector()

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
        y.extend(density_list_1)

    # 转换数据为PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # 定义模型
    input_size = X.shape[1]
    model = SimpleNet(input_size)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 训练模型
    train(model, X_tensor, y_tensor, optimizer, criterion, epochs=1000)



