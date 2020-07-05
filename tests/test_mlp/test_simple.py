import numpy as np
from biomed.mlp.simple import SimpleFFN
from biomed.properties_manager import PropertiesManager

def test_train_and_run_simple_ffn(datadir):
    X_test = np.load(datadir / 'X_test.npy')
    X_train = np.load(datadir / 'X_train.npy')
    Y_train = np.load(datadir / 'Y_train.npy')
    Y_test = np.load(datadir / 'Y_test.npy')
    input_dim = X_train.shape[1]
    nb_classes = len(Y_train[0])

    pm = PropertiesManager()
    pm.training_properties[ "epochs" ] = 1

    mlpsm = SimpleFFN.Factory.getInstance( pm )
    mlpsm.build_mlp_model(input_dim, nb_classes)
    predictions, scores = mlpsm.train_and_run_mlp_model(X_train, X_test, Y_train, Y_test)
    assert len(predictions) == len(Y_test)
