import csv
import os
from pathlib import Path

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
            '0_precision',
            '0_recall',
            '1_precision',
            '1_recall',
            'f1_micro_avg',
            'f1_macro_avg',
            'f1_weighted_avg']]


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
                if not csv_row[0]:
                    continue
                if csv_row[0] in ('0', '1'):
                    row.append(csv_row[1])
                    row.append(csv_row[2])
                else:
                    row.append(csv_row[3])
    return row


if __name__ == '__main__':
    results_directory = '/Users/riherrmann/Downloads/results'
    Path(results_directory + '/plots/').mkdir(parents=True, exist_ok=True)
    test_info_table = get_test_info_table_header()
    for test_name in os.listdir(results_directory):
        if '2020' not in test_name:
            continue
        print(test_name)
        # test_name = 'laniyd2-2020-09-30_03-00-15'
        test_info_row = get_test_info_table_row(test_name)
        test_info_table.append(test_info_row) if test_info_row else None
        create_features_scatterplot(test_name)
        create_history_plot(test_name)
    with open(f"{results_directory}/test_info_table.csv", 'w+') as file:
        csv.writer(file).writerows(test_info_table)
