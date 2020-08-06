from biomed.splitter.splitter import Splitter, SplitterFactory
from biomed.properties_manager import PropertiesManager
import biomed.services as Services
from sklearn.model_selection import train_test_split as simpleSplit
from sklearn.model_selection import StratifiedShuffleSplit as ComplexSplitter
from pandas import Series
from numpy import array as Array

class StdSplitter( Splitter ):
    def __init__( self, Properties: PropertiesManager ):
        self.__Properties = Properties

    def __mapEntry( self, Indices: Array, X: Series ) -> Series:
        Entry = []
        for Index in list( Indices ):
            Entry.append( X[ X.index[ Index ] ] )

        return Series( Entry )

    def __remap( self, BagOfIndices: list, X: Series ) -> list:
        Remaped = []
        for Indices in BagOfIndices:
            Remaped.append( (
                    self.__mapEntry( Indices[ 0 ], X ),
                    self.__mapEntry( Indices[ 1 ], X )
            ) )

        return Remaped

    def __splitFolds( self,  X: Series, Y: Series ) -> list:
        return self.__remap(
            ComplexSplitter(
                n_splits = self.__Properties.splitting[ 'folds' ],
                test_size = self.__Properties.splitting[ 'test' ],
                random_state = self.__Properties.splitting[ 'seed' ]
            ).split( X, Y ),
            X
        )

    def __split( self, X: Series, Y: Series, SplitKey: str ) -> tuple:
        return tuple( simpleSplit(
            X,
            test_size = self.__Properties.splitting[ SplitKey ],
            random_state = self.__Properties.splitting[ 'seed' ],
            shuffle = True,
            stratify = Y
        ) )

    def trainingSplit( self, X: Series, Y: Series ) -> list:
        if self.__Properties.splitting[ 'folds' ] == 1:
            return [ self.__split( X, Y, 'test' ) ]
        else:
            return self.__splitFolds( X, Y )

    def validationSplit( self, X: Series, Y: Series ) -> tuple:
        return self.__split( X, Y, 'validation' )

    class Factory( SplitterFactory ):
        @staticmethod
        def getInstance() -> Splitter:
            return StdSplitter(
                Services.getService( 'properties', PropertiesManager )
            )
