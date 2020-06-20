from normalizer import Normalizer
from normalizer import NormalizerFactory
import stanza

class ComplexNormalizer( Normalizer ):
    def __init__( self, Pipe ):
        self.__Pipe = Pipe

    def apply( self, Text: str, Flags: str ) -> list:
        return self.__filter( self.__Pipe( Text ).sentences[ 0 ].words, Flags )

    def __filter( self, Words: list, Flags ) -> list:
        Result = list()
        for Word in Words:
            if "n" in Flags:
                self.__appendNouns( Result, Word )
            if "v" in Flags:
                self.__appendVerbs( Result, Word )

        return Result

    def __appendNouns( self, Filtered: list, Word ):
        if Word.xpos == "NN" or Word.xpos == "NNP":
            Filtered.append( Word.lemma )

    def __appendVerbs( self, Filtered: list, Word ):
        if Word.upos == "VERB":
            Filtered.append( Word.lemma )

    class Factory( NormalizerFactory ):
        __Pipeline = stanza.Pipeline(
            lang='en',
            processors='tokenize,mwt,pos,lemma,ner'
        )

        def getInstance() -> Normalizer:
            return ComplexNormalizer( ComplexNormalizer.Factory.__Pipeline )
