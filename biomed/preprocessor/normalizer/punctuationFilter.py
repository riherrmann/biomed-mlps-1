from biomed.preprocessor.normalizer.filter import Filter
from biomed.preprocessor.normalizer.filter import FilterFactory
import string

class PunctuationFilter( Filter ):
    def __init__( self, Transition ):
        self.__Transition = Transition

    def apply( self, Text: str ) -> str:
        return Text.translate( self.__Transition )

    class Factory( FilterFactory ):
        __Transition = str.maketrans( '', '', string.punctuation )
        @staticmethod
        def getInstance() -> Filter:
            return PunctuationFilter( PunctuationFilter.Factory.__Transition )
