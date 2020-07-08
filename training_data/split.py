import csv
import numpy as np

input_data = list()

with open('train_75.tsv', 'r') as file:
    read_tsv = csv.reader(file, delimiter='\t')
    for row in read_tsv:
        input_data.append(row)

    y_test_doid, y_test_binary = list(), list()
    for row in input_data:
        y_test_doid.append(row[2])
        y_test_binary.append(row[3])
    print(y_test_doid)
    # np.save('Y_test_75_multi.npy', y_test_doid)
    # np.save('Y_test_75_binary.npy', y_test_binary)

# with open('train_25.tsv', "w", newline='') as file:
#     tsvwriter = csv.writer(file, delimiter='\t')
#     tsvwriter.writerows(input_data[0:int(len(input_data) / 4)])
#
# with open('train_75.tsv', "w", newline='') as file:
#     tsvwriter = csv.writer(file, delimiter='\t')
#     tsvwriter.writerow(input_data[0])
#     tsvwriter.writerows(input_data[int(len(input_data) / 4):])
