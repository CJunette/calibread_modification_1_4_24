import pandas as pd
from matplotlib import pyplot as plt
import sys
import os


def compare_error_of_different_model_input():
    df = pd.read_csv("data/error_of_different_model_input")
    task_index_list = df["task_index"].unique()

    task_index_list = [task_index_list[i] for i in range(len(task_index_list)) if task_index_list[i] != 11]
    task_index_list.sort()

    df_baseline = df[df["task_index"] == 11]
    x_baseline_list = [0]
    mean_baseline_list = [df_baseline["error"].mean()]
    std_baseline_list = [df_baseline["error"].std()]

    print(f"{mean_baseline_list[0]:.4f}, {std_baseline_list[0]:.4f}")

    x_list = [i + 1 for i in range(len(task_index_list))]
    mean_list = []
    std_list = []
    for task_index in task_index_list:
        df_task = df[df["task_index"] == task_index]
        mean_list.append(df_task["error"].mean())
        std_list.append(df_task["error"].std())
        print(f"{mean_list[-1]:.4f}, {std_list[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xticks([0] + x_list)

    ax.set_xlabel("Task Index")
    ax.set_ylabel("Accuracy Error Error")
    ax.set_title("Accuracy Error of Different Model Input")
    marker_size = 15
    line_width = 5
    ax.errorbar(x_baseline_list, mean_baseline_list, yerr=std_baseline_list, fmt='o', color=[0.9, 0.5, 0.2], markersize=marker_size, linewidth=line_width)
    ax.errorbar(x_list, mean_list, yerr=std_list, fmt='o', color=[0.3, 0.25, 0.7], markersize=marker_size, linewidth=line_width)

    plt.show()

    # TODO 未完成。





if __name__ == '__main__':
    compare_error_of_different_model_input()
