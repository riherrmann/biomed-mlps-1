from biomed import file_handler
from biomed.file_handler import FileHandler

if __name__ == '__main__':
    training_data_location = "training_data/train.tsv"
    file_handler = FileHandler()
    training_data = file_handler.read_tsv_pandas_data_structure(training_data_location)
    print(type(training_data))
    # for row in training_data['pmid']:
    #     print(row)
