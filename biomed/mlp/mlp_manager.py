from biomed.mlp.mlp import MLP
from biomed.mlp.mlp import MLPFactory
from biomed.mlp.input_data import InputData
import biomed.services as Services
from biomed.mlp.multiSimple import MultiSimpleFFN
from biomed.mlp.multiSimpleB import MultiSimpleBFFN
from biomed.mlp.simple import SimpleFFN
from biomed.mlp.simpleEx import SimpleExtendedFFN
from biomed.mlp.simpleB import SimpleBFFN
from biomed.mlp.simpleBEx import SimpleBExtendedFFN
from biomed.mlp.simpleC import SimpleCFFN
from biomed.mlp.simpleCEx import SimpleCExtendedFFN
from biomed.mlp.complex import ComplexFFN
from biomed.properties_manager import PropertiesManager
from numpy import array as Array

class MLPManager( MLP ):
    def __init__( self, PM: PropertiesManager ):
        self.__Models = {
            "s": SimpleFFN,
            "sx": SimpleExtendedFFN,
            "sb": SimpleBFFN,
            "sbx": SimpleBExtendedFFN,
            "sc": SimpleCFFN,
            "scx": SimpleCExtendedFFN,
            "ms": MultiSimpleFFN,
            "msb": MultiSimpleBFFN,
            "c": ComplexFFN,
        }

        self.__Model = None
        self.__Properties = PM

    def buildModel( self, Dimensions: int ) -> str:
        self.__Model = self.__Models[ self.__Properties.model ]( self.__Properties )
        return self.__Model.buildModel( Dimensions )

    def train( self, X: InputData, Y: InputData ) -> dict:
        return self.__Model.train( X, Y )

    def getTrainingScore( self, X: InputData, Y: InputData ) -> dict:
        return self.__Model.getTrainingScore( X, Y )

    def predict( self, ToPredict: tuple ) -> Array:
        return self.__Model.predict( ToPredict )

    class Factory( MLPFactory ):
        @staticmethod
        def getInstance():
            return MLPManager( PM = Services.getService( "properties", PropertiesManager ) )
