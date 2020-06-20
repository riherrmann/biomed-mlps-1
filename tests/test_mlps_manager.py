import numpy as np
from biomed.mlps_manager import MLPsManager
from biomed.properties_manager import PropertiesManager


def test_train_and_run_binary_mlp(datadir):
    X_test = np.load(datadir / 'X_test.npy')
    X_train = np.load(datadir / 'X_train.npy')
    Y_train = np.load(datadir / 'Y_train.npy')
    Y_test = np.load(datadir / 'Y_test.npy')
    input_dim = X_train.shape[1]
    nb_classes = len(Y_train[0])
    pm = PropertiesManager()
    mlpsm = MLPsManager(pm)
    mlpsm.build_binary_mlp(input_dim, nb_classes)
    predictions = mlpsm.train_and_run_binary_mlp(X_train, X_test, Y_train)
    assert len(predictions) == len(Y_test)
