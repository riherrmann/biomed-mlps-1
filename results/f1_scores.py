import csv
import glob
from sklearn.metrics import f1_score

import numpy as np

results = list()


class F1ScoreGetter:
    def __init__(self):
        self.Y_test_75_binary, self.Y_test_75_multi = self.get_y_data()

    def get_y_data(self):
        Y_test_75_binary = np.load('../training_data/Y_test_75_binary.npy')
        Y_test_75_multi = np.load('../training_data/Y_test_75_multi.npy')
        return Y_test_75_binary, Y_test_75_multi

    def get_preds_from_csv_file(self, file):
        input_data = self.get_list_from_csv(file)
        preds = list()
        for row in input_data:
            preds.append(row[1])
        preds_target = self.Y_test_75_multi if preds[0] == "doid" else self.Y_test_75_binary
        preds = preds[1:]
        # print(len(self.Y_test_75_binary))
        # print(len(preds))
        return preds, preds_target

    def get_list_from_csv(self, file, delimiter=','):
        input_data = list()
        read_csv = csv.reader(file, delimiter=delimiter)
        for row in read_csv:
            input_data.append(row)
        return input_data


if __name__ == '__main__':
    f1 = F1ScoreGetter()
    for file in glob.glob("*.csv*"):
        with open(file, 'r') as file:
            preds, preds_target = f1.get_preds_from_csv_file(file)
            if len(preds) == len(preds_target):
                score = f1_score(preds_target, preds, average='macro')
                print(file, score)
