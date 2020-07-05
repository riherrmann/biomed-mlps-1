from biomed.mlp.mlp import MLP
from biomed.mlp.multiSimple import MultiSimpleFFN
from biomed.mlp.simple import SimpleFFN
from biomed.mlp.simpleEx import SimpleExtendedFFN
from biomed.mlp.simpleB import SimpleBFFN
from biomed.mlp.simpleBEx import SimpleBExtendedFFN
from biomed.properties_manager import PropertiesManager

class MLPManager(MLP):
    __Models = {
        "s": SimpleFFN.Factory,
        "sx": SimpleExtendedFFN.Factory,
        "sb": SimpleBFFN.Factory,
        "sxb": SimpleBExtendedFFN.Factory,
        "ms": MultiSimpleFFN.Factory,
    }

    def __init__( self, pm: PropertiesManager ):
        self.__Model = MLPManager.__Models[ pm.model ].getInstance( pm )

    def build_mlp_model( self, input_dim, nb_classes ):
        self.__Model.build_mlp_model( input_dim, nb_classes )

    def train_and_run_mlp_model( self, X_train, X_test, Y_train, Y_test ):
        return self.__Model.train_and_run_mlp_model(X_train, X_test, Y_train, Y_test )
