from biomed.vectorizer.selector.selector import Selector
from biomed.properties_manager import PropertiesManager
from abc import abstractmethod
from pandas import Series
from numpy import array as Array

class SelectorBase( Selector ):
    def __init__( self, Properties: PropertiesManager ):
        self._Selector = None
        self._Properties = Properties

    @abstractmethod
    def _assembleSelector( self ):
        pass

    def build( self, X, Y: Series ):
        self._assembleSelector()
        self._Selector.fit( X, Y )

    def select( self, X: Array ) -> Array:
        self._validateSelector()
        return self._Selector.transform( X ).toarray()

    def _validateSelector( self ):
        if self._Selector is None:
            raise RuntimeError( "The selector must be builded before using it" )

    def _filterFeatureNamesByIndex(
        self,
        FeatureNames: list,
        IndiciesOfSelectedFeatures: list
    ) -> list:
        return [ FeatureNames[ I ] for I in IndiciesOfSelectedFeatures ]
