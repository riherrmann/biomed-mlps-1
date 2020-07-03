from biomed.mlp.mlp import MLP
from biomed.mlp.simple import SimpleFFN
from biomed.properties_manager import PropertiesManager


class MLPManager(MLP):
    def __init__(self, properties_manager: PropertiesManager):
        self.__model = SimpleFFN.Factory.getInstance( properties_manager )

    def build_mlp_model(self, input_dim, nb_classes):
        self.__model.build_mlp_model( input_dim, nb_classes )

    def train_and_run_mlp_model(self, X_train, X_test, Y_train):
        return self.__model.train_and_run_mlp_model(X_train, X_test, Y_train)
