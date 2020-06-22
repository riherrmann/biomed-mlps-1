from biomed.preprocessor.normalizer.filter import Filter
from biomed.preprocessor.normalizer.filter import FilterFactory
from nltk.stem import PorterStemmer

class StemFilter( Filter ):
    def __init__( self, Stemmer ):
        self.__Stemmer = Stemmer

    def apply( self, Text: str ) -> str:
        return self.__Stemmer.stem( Text )

    class Factory( FilterFactory ):
        @staticmethod
        def getInstance() -> Filter:
            return StemFilter( PorterStemmer() )
