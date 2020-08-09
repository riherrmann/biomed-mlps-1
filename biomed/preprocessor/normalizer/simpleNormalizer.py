from biomed.preprocessor.normalizer.normalizer import Normalizer
from biomed.preprocessor.normalizer.normalizer import NormalizerFactory
from biomed.preprocessor.normalizer.stemFilter import StemFilter
from biomed.preprocessor.normalizer.stopWordsFilter import StopWordsFilter
from biomed.preprocessor.normalizer.lowerFilter import LowerFilter
from biomed.preprocessor.normalizer.punctuationFilter import PunctuationFilter
from nltk import word_tokenize, sent_tokenize

class SimpleNormalizer( Normalizer ):
    def __init__( self, Filter ):
        self.__Filter = Filter

    def apply( self, StackOfDocuments: list, Flags: str ) -> list:
        ParsedStack = list()
        for Document in StackOfDocuments:
            ParsedStack.append( self.__filterDocument( Document, Flags ) )

        return ParsedStack

    def __filterDocument( self, Document: str, Flags: str ) -> str:
        return self._reassemble(
            self.__filterSentences( sent_tokenize( Document ), Flags )
        )

    def __filterSentences( self, Sentences: list, Flags: str ) -> str:
        ParsedSentences = list()
        for Sentence in Sentences:
            ParsedSentences.append(
                self._reassemble(
                    self.__applyFilters( word_tokenize( Sentence ), Flags )
                )
            )

        return ParsedSentences

    def __applyFilters( self, Tokens: list, Flags: str ) -> list:
        Purge = 0

        for Index in range( 0, len( Tokens ) ):
            Tokens[ Index ] = self.__filter( Tokens[ Index ], Flags )
            if not Tokens[ Index ]:
                Purge += 1

        self.__removeEmptyItems( Tokens, Purge )
        return Tokens

    def __filter( self, Token, Flags ) -> str:
        FilteredToken = Token

        if "l" in Flags:
            FilteredToken = self.__Filter[ "l" ].apply( FilteredToken )

        if "w" in Flags:
            FilteredToken = self.__Filter[ "w" ].apply( FilteredToken )

        if "s" in Flags:
            FilteredToken = self.__Filter[ "s" ].apply( FilteredToken )

        FilteredToken = self.__Filter[ "*" ].apply( FilteredToken )

        return FilteredToken

    def __removeEmptyItems( self, Tokens: list, Counter: int ):
        while Counter > 0:
            Tokens.remove( '' )
            Counter -= 1

    class Factory( NormalizerFactory ):
        __ApplicableFilter = {
            "s": StemFilter.Factory.getInstance(),
            "w": StopWordsFilter.Factory.getInstance(),
            "l": LowerFilter.Factory.getInstance(),
            "*": PunctuationFilter.Factory.getInstance()
        }

        def getApplicableFlags( self ) -> list:
            return [ "l", "s", "w" ]

        def getInstance( self ) -> Normalizer:
            return SimpleNormalizer( {
                    "s": StemFilter.Factory.getInstance(),
                    "w": StopWordsFilter.Factory.getInstance(),
                    "l": LowerFilter.Factory.getInstance(),
                    "*": PunctuationFilter.Factory.getInstance()
            } )
