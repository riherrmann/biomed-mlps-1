from abc import ABC, abstractmethod
from biomed.mlp.input_data import InputData
from numpy import array as Array

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True

class MLP( ABC ):
    @abstractmethod
    def buildModel( self, Dimension ) -> str:
        pass

    @abstractmethod
    def train( self, X: InputData, Y: InputData ) -> dict:
        pass

    @abstractmethod
    def getTrainingScore( self, X: InputData, Y: InputData ) -> dict:
        pass

    @abstractmethod
    def predict( self, ToPredict: Array ) -> Array:
        pass

class MLPFactory:
    @abstractstatic
    def getInstance() -> MLP:
        pass
