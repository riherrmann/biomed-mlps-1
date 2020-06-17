import csv
import pandas as pd


class FileHandler:
    def __init__(self):
        pass


    def read_tsv_pandas_data_structure(self, location):
        data = pd.read_csv(location, delimiter="\t")
        return data


    def open_tsv_as_list(self, location):
        output = []
        with open(location) as tsv_file:
            read_tsv = csv.reader(tsv_file, delimiter="\t")
            for row in read_tsv:
                output.append(row)
        return output


    def open_tsv_as_pubmed_dict(self, location):
        # pubmed_entries
        with open(location) as tsv_file:
            read_tsv = csv.DictReader(tsv_file, delimiter="\t")
        return read_tsv
