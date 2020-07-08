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
        preds = preds[1:] if preds[0] == 'doid' else self.convert_buggy_doid_to_real_binary(preds[1:])
        # print(f"{file} len(y) {len(self.Y_test_75_binary)} len(preds) {len(preds)}")
        return preds, preds_target

    def get_list_from_csv(self, file, delimiter=','):
        input_data = list()
        read_csv = csv.reader(file, delimiter=delimiter)
        for row in read_csv:
            input_data.append(row)
        return input_data

    def convert_buggy_doid_to_real_binary(self, preds):
        for i, entry in enumerate(preds):
            preds[i] = '0' if entry == '-1' else '1'
        return preds

if __name__ == '__main__':
    f1 = F1ScoreGetter()
    for file in glob.glob("*.csv*"):
        with open(file, 'r') as file:
            preds, preds_target = f1.get_preds_from_csv_file(file)
            if len(preds) == len(preds_target):
                matches, matches_pos, matches_neg = 0, 0, 0
                for i in range(len(preds)):
                    if preds[i] == preds_target[i]:
                        if preds[i] not in ('0', '-1'):
                            # print(preds[i], preds_target[i])
                            matches += 1
                    # else:
                    #     print(preds[i], preds_target[i])
                score = f1_score(preds_target, preds, average='macro')
                print(file.name, 'f1 score:', score, 'correctly predicted:', matches)
