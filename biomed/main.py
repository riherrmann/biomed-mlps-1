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
    print('Setup for input data')
    tmm.setup_for_input_data(training_data)
    target_dimension = 'doid'
    print('Setup for target dimension', target_dimension)
    tmm.setup_for_target_dimension(target_dimension)
    print('Build MLP and get predictions')
    preds = tmm.get_binary_mlp_predictions()
    print(preds)
    print('number of cancer predictions found:', len([x for x in preds if x != 0]))
    for x in preds:
        if x != 0:
            print('cancer found:', x)
