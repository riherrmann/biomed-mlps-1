import csv


class FileHandler:
    def __init__(self):
        pass

    def open_tsv(self, location):
        tsv_file = open(location)
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        tsv_file.close()
        return read_tsv
