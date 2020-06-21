from biomed.preprocessor.normalizer.filter import Filter
from biomed.preprocessor.normalizer.filter import FilterFactory
from nltk.corpus import stopwords

class StopWordsFilter( Filter ):
    def __init__( self, Words: list ):
        self.__Words = Words

    def apply( self, Text: str ) -> str:
        return "" if Text.lower() in self.__Words else Text

    class Factory( FilterFactory ):
        #We can customize that
        __Words = stopwords.words( 'english' )
        @staticmethod
        def getInstance() -> Filter:
            return StopWordsFilter( StopWordsFilter.Factory.__Words )
