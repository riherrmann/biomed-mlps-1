from abc import ABC, abstractmethod
from biomed.properties_manager import PropertiesManager
import numpy as np

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class MLP( ABC ):
    def __init__( self, Properties: PropertiesManager ):
        self._Properties = Properties
        self._Model = None

    @abstractmethod
    def build_mlp_model(self, input_dim, nb_classes):
        pass

    def __isMultiprocessing( self ):
        return True if self._Properties[ "training_properties" ][ "workers" ] > 1 else False

    def train_and_run_mlp_model(self, X_train, X_test, Y_train, Y_test):
        print("Training...")
        self._Model.fit(
            x = X_train,
            y = Y_train,
            epochs = self._Properties.training_properties[ 'epochs' ],
            batch_size = self._Properties.training_properties['batch_size'],
            validation_split = self._Properties.training_properties['validation_split'],
            workers = self._Properties.training_properties['workers'],
            use_multiprocessing = self.__isMultiprocessing()
        )

        print("Generating test predictions...")
        if len( Y_train[0] ) > 2:
            Predictions = np.argmax(
                self._Model.predict(
                    X_test,
                    batch_size = 1,
                    workers = self._Properties[ "training_properties" ][ "workers" ],
                    use_multiprocessing = self.__isMultiprocessing()
                ),
                axis = -1
            )
        else:
            Predictions = self._Model.predict_classes(
                X_test,
                batch_size = 1,
            )

        #see: https://keras.io/api/models/model/#evaluate & https://keras.io/api/models/model_training_apis/
        print( self._Model.metrics_names )
        Scores = self._Model.evaluate(
            X_test,
            Y_test,
            batch_size = self._Properties.training_properties['batch_size'],
            workers = self._Properties.training_properties['workers'],
            use_multiprocessing = self.__isMultiprocessing(),
            verbose = 0,
            return_dict = True
        )

        return ( Predictions, Scores )

class MLPFactory:
    @abstractstatic
    def getInstance( ModelProperties: PropertiesManager ) -> MLP:
        pass
