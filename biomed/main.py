from biomed import file_handler
from biomed.file_handler import FileHandler

if __name__ == '__main__':
    training_data_location = "training_data/train.tsv"
    file_handler = FileHandler()
    training_data = file_handler.open_tsv(training_data_location)
