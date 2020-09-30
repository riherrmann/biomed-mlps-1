import csv

import numpy as np
import matplotlib.pyplot as plt
import yaml


def get_feature_matrices_from_csv(file):
    x_list, y_list = list(), list()
    with open(file) as csvfile:
        for article in csv.reader(csvfile):
            if article[0] == '':
                continue
            for i, feature_value in enumerate(article):
                if i != 0 and float(feature_value) > 0.0:
                    x_list.append(int(i))
                    y_list.append(float(feature_value))
    return np.array(x_list), np.array(y_list)


def get_history_matrices_from_csv(file):
    labels = list()
    values = [list() for _ in range(4)]
    with open(file) as csvfile:
        for line in csv.reader(csvfile):
            if not labels:
                labels = line
            else:
                for i, val in enumerate(line):
                    values[i].append(float(val))

    return labels, values


def create_features_scatterplot(test_name):
    # Data
    file_path_training = f"{results_directory}/{test_name}/trainingFeatures.csv"
    file_path_test = f"{results_directory}/{test_name}/testFeatures.csv"
    data = (get_feature_matrices_from_csv(file_path_training), get_feature_matrices_from_csv(file_path_test))
    colors = ("red", "green")
    groups = ("training", "test")
    area = np.pi * 2

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.6, c=color, s=area, edgecolors='none', label=group)
    plt.title("Scatter plot")
    plt.xlabel('feature')
    plt.ylabel('tf-idf value')
    plt.legend(loc=2)
    plt.savefig(f"{results_directory}/{test_name}.png")
    plt.show()


def create_history_plot(test_name):
    file_path = f"{results_directory}/{test_name}/trainingHistory.csv"
    with open(f"{results_directory}/{test_name}/config.json") as file:
        config = yaml.safe_load(file)
        batch_size = config['training']['batch_size']
    labels, data = get_history_matrices_from_csv(file_path)
    y_max_acc = max(data[3])
    x_max_acc = data[3].index(y_max_acc)
    x = range(0, len(data[0]))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(labels)):
        ax.plot(x, data[i], label=labels[i])
    plt.annotate('max val_accuracy', xy=(x_max_acc, y_max_acc),
                 xytext=(x_max_acc + 5, y_max_acc - 0.25),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )
    plt.title(f"History plot: {test_name.split('-')[0]} (batch size: {batch_size})")
    plt.legend(loc=2)
    plt.savefig(f"{results_directory}/{test_name}/history.png")
    plt.show()


if __name__ == '__main__':
    results_directory = '/Users/riherrmann/Downloads/results'
    test_name = 'laniyd2-2020-09-30_03-00-15'
    # create_features_scatterplot(test_name)
    create_history_plot(test_name)
