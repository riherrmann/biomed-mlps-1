from pandas import DataFrame
from biomed.properties_manager import PropertiesManager
from biomed.preprocessor.polymorph_preprocessor import PolymorphPreprocessor
from biomed.text_mining_manager import TextMiningManager

class Pipeline:
    class Factory:
        @staticmethod
        def getInstance():
            return Pipeline( PropertiesManager() )

    def __init__( self, Properties: PropertiesManager ):
        self.__Properties = Properties

    def __startTextminer( self ):
        self.__TextMining = TextMiningManager(
            self.__Properties,
            PolymorphPreprocessor.Factory.getInstance( self.__Properties )
        )

    def pipe( self, training_data: DataFrame, test_data: DataFrame, properties: dict = None ):
        self.__reassign( properties )
        self.__startTextminer()

        print( 'Setup for input data')
        self.__TextMining.setup_for_input_data( training_data, test_data )
        print( 'Setup for target dimension', self.__Properties[ "classifier" ] )
        self.__TextMining.setup_for_target_dimension( self.__Properties[ "classifier" ] )
        print( 'Build MLP and get predictions' )
        return self.__TextMining.get_mlp_predictions()

    def __reassign( self, New: dict ):
        if not New:
            return
        else:
            for Key in New:
                self.__Properties[ Key ] = New[ Key ]
