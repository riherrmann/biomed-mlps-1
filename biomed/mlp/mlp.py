from abc import ABC, abstractmethod
from tensorflow.keras.callbacks import History
from biomed.mlp.input_data import InputData
import numpy as NP

class MLP( ABC ):
    @abstractmethod
    def buildModel( self, input_dim, nb_classes ) -> str:
        pass

    @abstractmethod
    def train( self, X: InputData, Y: InputData ) -> History:
        pass

    @abstractmethod
    def getTrainingScore( self, X: InputData, Y: InputData ) -> dict:
        pass

    @abstractmethod
    def predict( self, X_test: tuple ) -> NP.array:
        pass

class MLPFactory( ABC ):
    @abstractmethod
    def getInstance() -> MLP:
        pass
