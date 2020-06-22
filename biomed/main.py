import collections

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
        PolymorphPreprocessor.Factory.getInstance( pm )
    )
    print('Setup for input data')
    tmm.setup_for_input_data(training_data)
    target_dimension = 'doid'
    # target_dimension = 'is_cancer'
    print('Setup for target dimension', target_dimension)
    tmm.setup_for_target_dimension(target_dimension)
    print('Build MLP and get predictions')
    preds = tmm.get_binary_mlp_predictions()
    print(preds)
    cancer_types_found = [x for x in preds if x != 0]
    cancer_types_found = tmm.map_doid_values_to_nonsequential(cancer_types_found)
    print('number of cancer predictions found:', len(cancer_types_found))
    counter = collections.Counter(cancer_types_found)
    print('(doid, count):', counter.most_common())
