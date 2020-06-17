from biomed.file_handler import FileHandler
from biomed.text_mining_manager import TextMiningManager


def test_train_test_split(datadir):
    fm = FileHandler()
    data = fm.read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    sut = TextMiningManager()
    training_data, test_data = sut.train_test_split(data)
    assert training_data.shape == (int(data.shape[0] * 0.7), 5)
    assert test_data.shape == (int(data.shape[0] * 0.3), 5) or test_data.shape == (int(data.shape[0] * 0.3) + 1, 5)

    # train, test = train_test_split(data, test_size = 0.3)


def test_tfidf_transformation(datadir):
    fm = FileHandler()
    data = fm.read_tsv_pandas_data_structure(datadir / "test_train.tsv")
    sut = TextMiningManager()
    training_data, test_data = sut.train_test_split(data)
    max_features = 10000
    training_features, test_features = sut.tfidf_transformation(training_data, test_data, max_features)

    assert training_features.shape[0] == int(training_data.shape[0]) and training_features.shape[0] <= max_features
    assert test_features.shape[0] == int(test_data.shape[0]) and test_features.shape[0] <= max_features
