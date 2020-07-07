import csv

input_data = list()

with open('train.tsv', 'r') as file:
    read_tsv = csv.reader(file, delimiter='\t')
    for row in read_tsv:
        input_data.append(row)

with open('train_25.tsv', "w", newline='') as file:
    tsvwriter = csv.writer(file, delimiter='\t')
    tsvwriter.writerows(input_data[0:int(len(input_data) / 4)])

with open('train_75.tsv', "w", newline='') as file:
    tsvwriter = csv.writer(file, delimiter='\t')
    tsvwriter.writerow(input_data[0])
    tsvwriter.writerows(input_data[int(len(input_data) / 4):])
