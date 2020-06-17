from biomed.file_handler import FileHandler
from biomed.text_mining_manager import TextMiningManager

def test_train_test_split():
    fm = FileHandler()
    data = fm.read_tsv_pandas_data_structure("../training_data/train.tsv")
    sut = TextMiningManager()
    training_data, test_data = sut.train_test_split(data)
    assert training_data.shape == (int(data.shape[0] * 0.7), 5)
    assert test_data.shape == (int(data.shape[0] * 0.3), 5) or test_data.shape == (int(data.shape[0] * 0.3) + 1, 5)

    # train, test = train_test_split(data, test_size = 0.3)