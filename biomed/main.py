from biomed.file_handler import FileHandler
from biomed.properties_manager import PropertiesManager
from biomed.preprocessor.polymorph_preprocessor import PolymorphPreprocessor
from biomed.pipeline import pipeline


if __name__ == '__main__':
    training_data_location = "training_data/train.tsv"
    fh = FileHandler()
    training_data = fh.read_tsv_pandas_data_structure(training_data_location)
    pm = PropertiesManager()
    pp = PolymorphPreprocessor.Factory.getInstance( pm )

    counter = pipeline( training_data, pm, pp )
    print('(doid, count):', counter.most_common())
