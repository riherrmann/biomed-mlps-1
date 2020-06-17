from biomed.file_handler import FileHandler
from biomed.properties_manager import PropertiesManager
from biomed.text_mining_manager import TextMiningManager

if __name__ == '__main__':
    training_data_location = "training_data/train.tsv"
    fh = FileHandler()
    training_data = fh.read_tsv_pandas_data_structure(training_data_location)
    pm = PropertiesManager()
    tmm = TextMiningManager(pm)
    print(type(training_data))
    # for row in training_data['pmid']:
    #     print(row)
