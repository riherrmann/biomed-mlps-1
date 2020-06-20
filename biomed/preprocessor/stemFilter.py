from filter import Filter
from filter import FilterFactory
from nltk.stem import PorterStemmer

class StemFilter( Filter ):
    def __init__( self, Stemmer ):
        self.__Stemmer = Stemmer

    def apply( self, Text: str ) -> str:
        return self.__Stemmer.stem( Text )

    class Factory( FilterFactory ):
        __Stemmer = PorterStemmer()

        def getInstance() -> Filter:
            return StemFilter( StemFilter.Factory.__Stemmer )
