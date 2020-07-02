from unittest.mock import MagicMock, patch
from biomed.file_handler import FileHandler
from biomed.properties_manager import PropertiesManager
from biomed.text_mining_manager import TextMiningManager
from biomed.mlp_manager import MLPManager
from biomed.preprocessor.pre_processor import PreProcessor
from pandas import DataFrame

class StubbedPreprocessor(PreProcessor):
    def __init__(self):
        self.WasCalled = False
        self.LastFlags = ""

    def preprocess_text_corpus(self, frame: DataFrame, flags: str) -> list:
        self.WasCalled = True
        self.LastFlags = flags
        return frame["text"]


def test_train_test_split(datadir):
    data = FileHandler().read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pp = StubbedPreprocessor()
    pm = PropertiesManager()
    sut = TextMiningManager(pm, pp)
    training_data, test_data = sut._data_train_test_split(data)
    assert training_data.shape == (int(data.shape[0] * (1 - pm.test_size)), 5)
    assert test_data.shape == (int(data.shape[0] * pm.test_size), 5) \
        or test_data.shape == (int(data.shape[0] * pm.test_size) + 1, 5)


def test_preprocessor(datadir):
    data = FileHandler().read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pm = PropertiesManager()
    pp = StubbedPreprocessor()
    pm.preprocessor_variant = "swl"
    sut = TextMiningManager(pm, pp)
    training_data, test_data = sut._data_train_test_split(data)
    training_features, test_features = sut._tfidf_transformation(training_data, test_data)
    assert pp.WasCalled == True
    assert pp.LastFlags == pm.preprocessor_variant


def test_tfidf_transformation(datadir):
    data = FileHandler().read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pp = StubbedPreprocessor()
    pm = PropertiesManager()
    sut = TextMiningManager(pm, pp)
    training_data, test_data = sut._data_train_test_split(data)
    max_features = 200000
    training_features, test_features = sut._tfidf_transformation(training_data, test_data)

    assert training_features.shape[0] == training_data.shape[0] and training_features.shape[1] <= max_features
    assert test_features.shape[0] == test_data.shape[0] and test_features.shape[1] <= max_features


def test_setup_for_input_data(datadir):
    data = FileHandler().read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pp = StubbedPreprocessor()
    pm = PropertiesManager()
    sut = TextMiningManager(pm, pp)
    sut.setup_for_input_data(data)
    assert sut.input_dim == sut.training_features.shape[1]  # in range(5000, 6000)


def test_prepare_input_data(datadir):
    data = FileHandler().read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pm = PropertiesManager()
    pp = StubbedPreprocessor()
    test_size = pm.test_size
    sut = TextMiningManager(pm, pp)
    sut._prepare_input_data(data)
    max_features = pm.tfidf_transformation_properties['max_features']
    assert int(data.shape[0] * test_size) <= sut.X_test.shape[0] <= int(data.shape[0] * test_size) + 1 and \
           sut.X_test.shape[1] <= max_features
    assert int(data.shape[0] * (1 - test_size)) <= sut.X_train.shape[0] <= int(
        data.shape[0] * (1 - test_size)) + 1 and sut.X_train.shape[1] <= max_features


def test_setup_for_target_dimension(datadir):
    data = FileHandler().read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pp = StubbedPreprocessor()
    pm = PropertiesManager()
    sut = TextMiningManager(pm, pp)
    sut.setup_for_input_data(data)
    sut.setup_for_target_dimension('is_cancer')
    assert sut.nb_classes == 2
    assert sut.Y_train.shape[1] == 2
    # numpy.save('tests/test_mlps_manager/X_train_binary.npy', sut.X_train)
    # numpy.save('tests/test_mlps_manager/X_test_binary.npy', sut.X_test)
    # numpy.save('tests/test_mlps_manager/Y_train_binary.npy', sut.Y_train)
    # numpy.save('tests/test_mlps_manager/Y_test_binary.npy', sut.Y_test)

    sut.setup_for_target_dimension('doid')
    assert sut.nb_classes == 8
    assert sut.Y_train.shape[1] == 8
    # numpy.save('tests/test_mlps_manager/X_train.npy', sut.X_train)
    # numpy.save('tests/test_mlps_manager/X_test.npy', sut.X_test)
    # numpy.save('tests/test_mlps_manager/Y_train.npy', sut.Y_train)
    # numpy.save('tests/test_mlps_manager/Y_test.npy', sut.Y_test)

def test_map_doid_values_to_sequential(datadir):
    pp = StubbedPreprocessor()
    pm = PropertiesManager()
    mlp = MagicMock( spec = MLPManager )
    mlp.train_and_run_mlp_model_1.return_value = [-1, 1234, 789, 42, -1]

    sut = TextMiningManager(pm, pp)
    sut.doid_unique = [-1, 1234, 789, 42]
    sut.mlpsm = mlp

    output_y_data = sut.get_binary_mlp_predictions( sequential = True )
    assert output_y_data[ 1 ] == [0, 1, 2, 3, 0]

def test_map_doid_values_to_nonsequential(datadir):
    pp = StubbedPreprocessor()
    pm = PropertiesManager()
    mlp = MagicMock( spec = MLPManager )
    mlp.train_and_run_mlp_model_1.return_value = [0, 1, 2, 3, 0]

    sut = TextMiningManager(pm, pp)
    sut.doid_unique = [-1, 1234, 789, 42]
    sut.mlpsm = mlp

    output_y_data = sut.get_binary_mlp_predictions( sequential = False )
    assert output_y_data[ 1 ] == [-1, 1234, 789, 42, -1]
