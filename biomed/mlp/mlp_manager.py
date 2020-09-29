from biomed.mlp.mlp import MLP
from biomed.mlp.mlp import MLPFactory
from biomed.mlp.input_data import InputData
import biomed.services_getter as ServiceGetter
from biomed.mlp.bin_tow_layered import Bin2Layered
from biomed.mlp.bin_tow_w_layered import WeightedBin2Layered
from biomed.mlp.bin_2_w_d_layered import WeightedBin2LayeredDrop
from biomed.mlp.bin_3_layered import Bin3Layered
from biomed.mlp.bin_3_w_layered import WeightedBin3Layered
from biomed.properties_manager import PropertiesManager
from typing import Union
from numpy import array as Array

class MLPManager( MLP ):
    def __init__( self, PM: PropertiesManager ):
        self.__Models = {
            "b2": Bin2Layered,
            "wb2": WeightedBin2Layered,
            "wb2d": WeightedBin2LayeredDrop,
            "b3": Bin3Layered,
            "wb3": WeightedBin3Layered,
        }

        self.__Model = None
        self.__Properties = PM

    def buildModel( self, Dimensions: int, Weights: Union[ None, dict ] = None ) -> str:
        self.__Model = self.__Models[ self.__Properties.model ]( self.__Properties )
        return self.__Model.buildModel( Dimensions, Weights )

    def train( self, X: InputData, Y: InputData ) -> dict:
        return self.__Model.train( X, Y  )

    def getTrainingScore( self, X: InputData, Y: InputData ) -> dict:
        return self.__Model.getTrainingScore( X, Y )

    def predict( self, ToPredict: tuple ) -> Array:
        return self.__Model.predict( ToPredict )

    class Factory( MLPFactory ):
        @staticmethod
        def getInstance( getService: ServiceGetter ):
            return MLPManager( getService( "properties", PropertiesManager ) )
