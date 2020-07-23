from biomed.properties_manager import PropertiesManager
from biomed.mlp.mlp import MLP
from biomed.mlp.input_data import InputData
from tensorflow.keras.callbacks import History
import numpy as NP

class ModelBase( MLP ):
    def __init__( self, Properties: PropertiesManager ):
        self._Properties = Properties
        self._Model = None
        self.__Dim = 0

    def _summarize( self ):
        Summery = []
        self._Model.summary( print_fn = lambda X: Summery.append( X ) )
        return "\n".join( Summery )

    def train( self, X: InputData, Y: InputData ) -> History:
        self.__Dim = len( Y.Train[0] )

        print("Training...")
        Hist = self._Model.fit(
            x = X.Train,
            y = Y.Train,
            shuffle = True,
            epochs = self._Properties.training[ 'epochs' ],
            batch_size = self._Properties.training['batch_size'],
            validation_data = ( X.Val, Y.Val ),
            workers = self._Properties.training['workers'],
            use_multiprocessing = self.__isMultiprocessing()
        )

        return Hist

    def __verifyTraining( self ):
        if not self.__Dim:
            raise RuntimeError( "The model has not be trained" )

    def getTrainingScore( self, X: InputData, Y: InputData ) -> dict:
        self.__verifyTraining()
        return self._Model.evaluate( X.Test, Y.Test, verbose = 0 )

    def __isMultiClass( self ) -> bool:
        return self.__Dim > 2

    def __isMultiprocessing( self ):
        return True if self._Properties.training[ "workers" ] > 1 else False

    def __predict( self, ToPredict: tuple ) -> NP.array:
        return self._Model.predict(
            ToPredict,
            batch_size = self._Properties.training['batch_size'],
            workers = self._Properties.training[ 'workers' ],
            use_multiprocessing = self.__isMultiprocessing()
        )

    def __normalizeBinary( self, Predictions: NP.array ) -> NP.array:
        return NP.where( Predictions < 0.5, 0, 1 )

    def __normalizeMulti( self, Predictions: NP.array ) -> NP.array:
        return NP.argmax( Predictions, axis = 1 )

    def predict( self, ToPredict: tuple ) -> NP.array:
        self.__verifyTraining()

        print("Generating test predictions...")
        if self.__isMultiClass():
            return self.__normalizeMulti( self.__predict( ToPredict ) )
        else:
            return self.__normalizeBinary( self.__predict( ToPredict ) )
