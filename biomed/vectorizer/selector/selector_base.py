from biomed.vectorizer.selector.selector import Selector
from abc import abstractmethod
from pandas import Series
from numpy import array as Array

class SelectorBase( Selector ):
    def __init__( self ):
        self._Selector = None

    @abstractmethod
    def _assembleSelector( self ):
        pass

    def build( self, X, Y: Series ):
        self._assembleSelector()
        self._Selector.fit( X, Y )

    def select( self, X: Array ) -> Array:
        if self._Selector is None:
            raise RuntimeError( "The selector must be builded before using it" )

        return self._Selector.transform( X ).toarray()
