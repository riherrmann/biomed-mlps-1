import numpy

from biomed.file_handler import FileHandler
from biomed.properties_manager import PropertiesManager
from biomed.text_mining_manager import TextMiningManager


def test_train_test_split(datadir):
    data = FileHandler().read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pm = PropertiesManager()
    sut = TextMiningManager(pm)
    training_data, test_data = sut._data_train_test_split(data)
    assert training_data.shape == (int(data.shape[0] * (1 - pm.test_size)), 5)
    assert test_data.shape == (int(data.shape[0] * pm.test_size), 5) \
        or test_data.shape == (int(data.shape[0] * pm.test_size) + 1, 5)


def test_tfidf_transformation(datadir):
    data = FileHandler().read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pm = PropertiesManager()
    sut = TextMiningManager(pm)
    training_data, test_data = sut._data_train_test_split(data)
    max_features = 200000
    training_features, test_features = sut._tfidf_transformation(training_data, test_data)

    assert training_features.shape[0] == training_data.shape[0] and training_features.shape[1] <= max_features
    assert test_features.shape[0] == test_data.shape[0] and test_features.shape[1] <= max_features


def test_setup_for_input_data(datadir):
    data = FileHandler().read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pm = PropertiesManager()
    sut = TextMiningManager(pm)
    sut.setup_for_input_data(data)
    assert sut.input_dim == sut.training_features.shape[1]  # in range(5000, 6000)


def test_prepare_input_data(datadir):
    data = FileHandler().read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pm = PropertiesManager()
    test_size = pm.test_size
    sut = TextMiningManager(pm)
    sut._prepare_input_data(data)
    max_features = pm.tfidf_transformation_properties['max_features']
    assert int(data.shape[0] * test_size) <= sut.X_test.shape[0] <= int(data.shape[0] * test_size) + 1 and \
        sut.X_test.shape[1] <= max_features
    assert int(data.shape[0] * (1 - test_size)) <= sut.X_train.shape[0] <= int(
        data.shape[0] * (1 - test_size)) + 1 and sut.X_train.shape[1] <= max_features


def test_setup_for_target_dimension(datadir):
    data = FileHandler().read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    pm = PropertiesManager()
    sut = TextMiningManager(pm)
    sut.setup_for_input_data(data)
    sut.setup_for_target_dimension('is_cancer')
    assert sut.nb_classes == 2
    sut.setup_for_target_dimension('doid')
    assert sut.nb_classes == 8
    assert sut.Y_train.shape[1] == 8
    # numpy.save('tests/test_mlps_manager/X_train.npy', sut.X_train)
    # numpy.save('tests/test_mlps_manager/X_test.npy', sut.X_test)
    # numpy.save('tests/test_mlps_manager/Y_train.npy', sut.Y_train)
    # numpy.save('tests/test_mlps_manager/Y_test.npy', sut.Y_test)


def test_map_doid_values_to_sequential(datadir):
    pm = PropertiesManager()
    sut = TextMiningManager(pm)
    sut.doid_unique = [-1, 1234, 789, 42]
    input_y_data = [-1, 1234, 789, 42, -1]
    output_y_data = sut.map_doid_values_to_sequential(input_y_data)
    assert output_y_data == [0, 1, 2, 3, 0]


def test_map_doid_values_to_nonsequential(datadir):
    pm = PropertiesManager()
    sut = TextMiningManager(pm)
    sut.doid_unique = [-1, 1234, 789, 42]
    input_y_data = [0, 1, 2, 3, 0]
    output_y_data = sut.map_doid_values_to_nonsequential(input_y_data)
    assert output_y_data == [-1, 1234, 789, 42, -1]
