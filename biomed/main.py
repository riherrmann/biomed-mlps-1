from biomed.file_handler import FileHandler
from biomed.properties_manager import PropertiesManager
from biomed.text_mining_manager import TextMiningManager
from biomed.preprocessor.polymorph_preprocessor import PolymorphPreprocessor

if __name__ == '__main__':
    training_data_location = "training_data/train.tsv"
    fh = FileHandler()
    training_data = fh.read_tsv_pandas_data_structure(training_data_location)
    pm = PropertiesManager()
    tmm = TextMiningManager(
        pm,
        PolymorphPreprocessor.Factory.getInstance()
    )
    tmm.setup_for_input_data(training_data)
    tmm.setup_for_target_dimension('is_cancer')
    preds = tmm.get_binary_mlp_predictions()
    print(preds)
    # for row in training_data['pmid']:
    #     print(row)
