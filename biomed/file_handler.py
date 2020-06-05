import csv


class FileHandler:
    def __init__(self):
        pass

    def open_tsv_as_list(self, location):
        output = []
        with open(location) as tsv_file:
            read_tsv = csv.reader(tsv_file, delimiter="\t")
            for row in read_tsv:
                output.append(row)
        return output


    def open_tsv_as_dict(self, location):
        with open(location) as tsv_file:
            read_tsv = csv.DictReader(tsv_file, delimiter="\t")
        return read_tsv
