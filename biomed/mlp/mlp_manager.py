from biomed.mlp.mlp import MLP
from biomed.mlp.mlp import MLPFactory
from biomed.mlp.input_data import InputData
import biomed.services_getter as ServiceGetter
from biomed.mlp.bin_tow_layered import Bin2Layered
from biomed.properties_manager import PropertiesManager
from numpy import array as Array

class MLPManager( MLP ):
    def __init__( self, PM: PropertiesManager ):
        self.__Models = {
            "b2": Bin2Layered,
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
        def getInstance( getService: ServiceGetter ):
            return MLPManager( getService( "properties", PropertiesManager ) )
