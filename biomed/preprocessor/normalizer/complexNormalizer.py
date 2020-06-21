from normalizer.normalizer import Normalizer
from normalizer.normalizer import NormalizerFactory
import stanza

class ComplexNormalizer( Normalizer ):
    def __init__( self, Pipe ):
        self.__Pipe = Pipe

    def apply( self, Text: str, Flags: str ) -> str:
        return self._reassemble(
            self.__filter( self.__Pipe( Text ).sentences[ 0 ].words, Flags )
        )

    def __filter( self, Words: list, Flags ) -> list:
        Result = list()
        for Word in Words:
            if "n" in Flags:
                self.__appendNouns( Result, Word )
            if "v" in Flags:
                self.__appendVerbs( Result, Word )
            if "a" in Flags:
                self.__appendAdj( Result, Word )


        return Result

    def __appendNouns( self, Filtered: list, Word ):
        if Word.xpos == "NN" or Word.xpos == "NNP":
            self.__append( Filtered, Word )

    def __appendVerbs( self, Filtered: list, Word ):
        if Word.upos == "VERB":
            self.__append( Filtered, Word )

    def __appendAdj( self, Filtered: list, Word ):
        if Word.upos == "ADJ":
            self.__append( Filtered, Word )

    def __append( self, Filtered: list, Word ):
        Filtered.append( Word.lemma )

    class Factory( NormalizerFactory ):
        __Pipeline = stanza.Pipeline(
            lang='en',
            processors='tokenize,mwt,pos,lemma,ner'
        )
        @staticmethod
        def getInstance() -> Normalizer:
            return ComplexNormalizer( ComplexNormalizer.Factory.__Pipeline )
