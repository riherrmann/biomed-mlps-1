from biomed.file_handler import FileHandler
from biomed.properties_manager import PropertiesManager
from biomed.text_mining_manager import TextMiningManager


def test_train_test_split(datadir):
    fm = FileHandler()
    data = fm.read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pm = PropertiesManager()
    sut = TextMiningManager(pm)
    training_data, test_data = sut._data_train_test_split(data)
    assert training_data.shape == (int(data.shape[0] * (1 - pm.test_size)), 5)
    assert test_data.shape == (int(data.shape[0] * pm.test_size), 5) \
        or test_data.shape == (int(data.shape[0] * pm.test_size) + 1, 5)


def test_tfidf_transformation(datadir):
    fm = FileHandler()
    data = fm.read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pm = PropertiesManager()
    sut = TextMiningManager(pm)
    training_data, test_data = sut._data_train_test_split(data)
    max_features = 200000
    training_features, test_features = sut._tfidf_transformation(training_data, test_data)

    assert training_features.shape[0] == training_data.shape[0] and training_features.shape[1] <= max_features
    assert test_features.shape[0] == test_data.shape[0] and test_features.shape[1] <= max_features


def test_setup_for_input_data(datadir):
    fm = FileHandler()
    data = fm.read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pm = PropertiesManager()
    sut = TextMiningManager(pm)
    sut.setup_for_input_data(data)
    assert sut.input_dim == sut.training_features.shape[1]


def test_prepare_input_data(datadir):
    fm = FileHandler()
    data = fm.read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pm = PropertiesManager()
    test_size = pm.test_size
    sut = TextMiningManager(pm)
    sut._prepare_input_data(data)
    max_features = pm.tfidf_transformation_properties['max_features']
    assert int(data.shape[0] * test_size) <= sut.X_test.shape[0] <= int(data.shape[0] * test_size) + 1 and \
           sut.X_test.shape[1] <= max_features
    assert int(data.shape[0] * (1 - test_size)) <= sut.X_train.shape[0] <= int(
        data.shape[0] * (1 - test_size)) + 1 and sut.X_train.shape[1] <= max_features
