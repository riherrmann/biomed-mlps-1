from pre_processor import PreProcessor
from pre_processor import PreProcessorFactory
from nltk import sent_tokenize
from normalizer.normalizer import Normalizer
from normalizer.simpleNormalizer import SimpleNormalizer
from normalizer.complexNormalizer import ComplexNormalizer

class PolymorphPreprocessor( PreProcessor ):
    def __init__(
        self,
        Simple: Normalizer,
        SimpleFlags: list,
        Complex: Normalizer,
        ComplexFlags: list
    ):
        self.__SimpleFlags = SimpleFlags
        self.__Simple = Simple
        self.__ComplexFlags = ComplexFlags
        self.__Complex = Complex

    def preprocess_text_corpus( self, text: str, flags: str ) -> str:
        return self.__reassemble( self.__normalize( sent_tokenize( text ), flags ) )

    def __normalize( self, Sentences: list, Flags: str ) -> list:
        ParsedSentences = list()
        for Sentence in Sentences:
            ParsedSentences.append( self.__normalizePerSentence( Sentence, Flags ) )

        return ParsedSentences

    def __normalizePerSentence( self, Text: str, Flags: str ) -> str:
        ParsedSentence = Text
        if self.__useComplex( Flags ):
            ParsedSentence = self.__reassemble( self.__Complex.apply( ParsedSentence, Flags ) )

        if self.__useSimple( Flags ):
            ParsedSentence = self.__reassemble( self.__Simple.apply( ParsedSentence, Flags ) )

        return ParsedSentence

    def __useSimple( self, Flags: list ) -> bool:
        for Flag in Flags:
            if Flag in self.__SimpleFlags:
                return True
        else:
            return False

    def __useComplex( self, Flags: list ) -> bool:
        for Flag in Flags:
            if Flag in self.__ComplexFlags:
                return True
        else:
            return False

    def __reassemble( self, Text: list ) -> str:
        return " ".join( Text )

    class Factory( PreProcessorFactory ):
        __Simple = SimpleNormalizer.Factory.getInstance()
        __SimpleFlags = [ "s", "l", "w" ]
        __Complex = ComplexNormalizer.Factory.getInstance()
        __ComplexFlags = [ "n", "v", "a" ]

        def getInstance() -> PreProcessor:
            return PolymorphPreprocessor(
                PolymorphPreprocessor.Factory.__Simple,
                PolymorphPreprocessor.Factory.__SimpleFlags,
                PolymorphPreprocessor.Factory.__Complex,
                PolymorphPreprocessor.Factory.__ComplexFlags
            )
