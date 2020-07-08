import csv
import numpy as np

input_data = list()

with open('train.tsv', 'r') as file:
    read_tsv = csv.reader(file, delimiter='\t')
    for row in read_tsv:
        input_data.append(row)

    y_test_doid, y_test_binary = list(), list()
    processed_pmids = list()
    for i, row in enumerate(input_data):
        if i == 0 or row[0] in processed_pmids:
            continue
        processed_pmids.append(row[0])
        y_test_doid.append(row[2])
        y_test_binary.append(row[3])
    print(y_test_binary)
    np.save('Y_test_75_multi.npy', y_test_doid)
    np.save('Y_test_75_binary.npy', y_test_binary)

# with open('train_25.tsv', "w", newline='') as file:
#     tsvwriter = csv.writer(file, delimiter='\t')

# with open('train_25.tsv', "w", newline='') as file:
#     tsvwriter = csv.writer(file, delimiter='\t')
#     tsvwriter.writerows(input_data[0:int(len(input_data) / 4)])
#
# with open('y_75_binary.csv', "w", newline='') as file:
#     tsvwriter = csv.writer(file, delimiter=',')
#     for i in range(len(processed_pmids)):
#         tsvwriter.writerow(processed_pmids[i], y_test_binary[i])
