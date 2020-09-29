import csv

import numpy as np
import matplotlib.pyplot as plt


def get_matrices_from_csv(file):
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


# Create data
test_directory = 'test1-2020-09-29_21-05-38/'
file_path_training = f"{test_directory}trainingFeatures.csv"
file_path_test = f"{test_directory}testFeatures.csv"
data = (get_matrices_from_csv(file_path_training), get_matrices_from_csv(file_path_test))
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
plt.show()
