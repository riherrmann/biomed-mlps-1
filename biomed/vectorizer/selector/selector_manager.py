from biomed.vectorizer.selector.selector import Selector
from biomed.vectorizer.selector.selector import SelectorFactory
from biomed.properties_manager import PropertiesManager
import biomed.services as Services
from pandas import Series
from numpy import array as Array

class SelectorManager( Selector ):
    def __init__( self, Properties: PropertiesManager ):
        self.__Properties = Properties
        self.__Selectors = {}

    def build( self, X: Array, Y: Series ):
        self.__Selector = None

    def select( self, X: Array ) -> Array:
        if not self.__Selector:
            return X.toarray()

    def getSupportedFeatures( self, Labels: list ) -> list:
        if not self.__Selector:
            return Labels

    class Factory( SelectorFactory ):
        @staticmethod
        def getInstance() -> Selector:
            return SelectorManager(
                Services.getService( "properties", PropertiesManager )
            )
