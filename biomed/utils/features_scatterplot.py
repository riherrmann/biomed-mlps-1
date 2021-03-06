import csv
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas
import yaml
from sklearn.metrics import confusion_matrix


def get_best_accuracy_loss_point(test_name):
    file_path = f"{results_directory}/{test_name}/trainingHistory.csv"
    if not os.path.isfile(file_path):
        return
    _, data = get_history_matrices_from_csv(file_path)
    max_val_accuracy = max(data[3])
    best_point = [0, 0, 0]  # (x, val_accuracy, val_acc - val_loss)
    for epoch in range(len(data[3])):
        acc_loss = data[3][epoch] - data[2][epoch]
        if data[3][epoch] >= 0.9 * max_val_accuracy and acc_loss > best_point[2]:
            best_point = [epoch, data[3][epoch], acc_loss]
    return best_point


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
    if not (os.path.isfile(file_path_training) and os.path.isfile(file_path_test)):
        return
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
    plt.savefig(f"{results_directory}/plots/features_{test_name}.png")
    # plt.show()
    plt.close()


def create_history_plot(test_name):
    file_path = f"{results_directory}/{test_name}/trainingHistory.csv"
    if not os.path.isfile(file_path):
        return
    config_path = f"{results_directory}/{test_name}/config.json"
    if os.path.isfile(config_path):
        with open(config_path) as file:
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
    plt.title(f"History plot: {test_name.split('-')[0]} (batch size: {batch_size if batch_size else 'N/A'})")
    plt.legend(loc=2)
    plt.savefig(f"{results_directory}/plots/history_{test_name}.png")
    # plt.show()
    plt.close()


def get_test_info_table_header():
    return [['test_name',
             'classifier',
             'preprocessing_variant',
             'model',
             'vectorizing_max_features',
             'vectorizing_ngram_range',
             'batch_size',
             'selection_type',
             'selection_features',
             'f1_micro_avg',
             'f1_macro_avg',
             'f1_weighted_avg',
             'precision',
             'recall']]


def get_test_info_table_row(test_name):
    row = list([test_name])
    config_path = f"{results_directory}/{test_name}/config.json"
    if os.path.isfile(config_path):
        with open(config_path) as file:
            config = yaml.safe_load(file)
            row.append(config.get('classifier'))
            row.append(config.get('preprocessing', dict()).get('variant'))
            row.append(config.get('model'))
            row.append(config.get('vectorizing', dict()).get('max_features'))
            row.append(config.get('vectorizing', dict()).get('ngram_range')[1])
            row.append(config.get('training', dict()).get('batch_size'))
            row.append(config.get('selection', dict()).get('type'))
            row.append(config.get('selection', dict()).get('amountOfFeatures'))

    class_report_path = f"{results_directory}/{test_name}/classReport.csv"
    if os.path.isfile(class_report_path):
        with open(class_report_path) as file:
            for csv_row in csv.reader(file):
                if 'avg' in csv_row[0]:
                    row.append(csv_row[3])
                    if csv_row[0] == 'weighted avg':
                        row.append(csv_row[1])
                        row.append(csv_row[2])
    return row


def generate_k_fold_heatmaps(test_name):
    test_dir = f"{results_directory}/{test_name}"
    predictions_path = f"{test_dir}/predictions.csv"
    if '.' in test_dir or not os.path.isfile(predictions_path):
        return
    csv_data = pandas.read_csv(predictions_path)
    cm = confusion_matrix(csv_data['predicted'], csv_data['actual'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    vmax = 3000 if results_directory.endswith('bin') else 20
    cax = ax.matshow(cm, vmin=0, vmax=vmax)
    fig.colorbar(cax)
    plt.title(test_name)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"{results_directory}/plots/heatmap_{test_name.replace('/', '_')}.png")
    # plt.show()
    plt.close()


if __name__ == '__main__':
    results_directory = '/Users/riherrmann/Downloads/results/doid'
    Path(results_directory + '/plots/').mkdir(parents=True, exist_ok=True)
    test_info_table = get_test_info_table_header()
    acc_loss = [['name', 'x', 'val_accuracy', 'acc_minus_loss']]
    for test_name in os.listdir(results_directory):
        if '2020' not in test_name:
            continue
        print(test_name)

        if test_name.startswith('fold_'):
            for i in os.listdir(f"{results_directory}/{test_name}"):
                generate_k_fold_heatmaps(f"{test_name}/{i}")
        else:
            create_features_scatterplot(test_name)
            create_history_plot(test_name)
            generate_k_fold_heatmaps(test_name)

            test_info_row = get_test_info_table_row(test_name)
            test_info_table.append(test_info_row) if test_info_row else None
            acc_loss_test = get_best_accuracy_loss_point(test_name)
            acc_loss.append([test_name] + acc_loss_test) if acc_loss_test else None

    with open(f"{results_directory}/test_info_table.csv", 'w+') as file:
        csv.writer(file).writerows(test_info_table)
    with open(f"{results_directory}/acc_loss_table.csv", 'w+') as file:
        csv.writer(file).writerows(acc_loss)
