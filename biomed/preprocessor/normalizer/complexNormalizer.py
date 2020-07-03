from biomed.preprocessor.normalizer.normalizer import Normalizer
from biomed.preprocessor.normalizer.normalizer import NormalizerFactory
import re as RegEx
from stanza import Document
from stanza.models.common.doc import Word as NLPToken
import stanza

class ComplexNormalizer( Normalizer ):
    __Pattern = RegEx.compile( r'\n+' )

    def __init__( self ):
        self.__Pipe = None

    def __initPipeLine( self ):
        self.__Pipe = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,lemma',
            use_gpu=True,
            logging_level = "WARN",
            verbose=False,
            tokenize_batch_size=128,
            lemma_batch_size=200,
        )

    def apply( self, StackOfDocument: list, Flags: str ) -> list:
        self.__initPipeLine()
        return self.__batchAndApply( StackOfDocument, Flags )

    def __batchAndApply( self, StackOfDocument: list, Flags: str ) -> list:
        Batch = self.__glueDocumentsTogether( StackOfDocument )
        return self.__filterSentencesAndSplitDocuments(
            self.__Pipe( Batch[ 0 ] ),
            Batch[ 1 ],
            Flags
        )

    def __glueDocumentsTogether( self, StackOfDocument: list ) -> tuple:
        StackOfDocument = list( StackOfDocument )
        EndOfDocuments = list()

        StackOfDocument[ 0 ] = self.__adjustFormat( StackOfDocument[ 0 ] )
        EndOfDocuments.append( len( StackOfDocument[ 0 ] ) )

        for Index in range( 1, len( StackOfDocument ) ):
            StackOfDocument[ Index ] = self.__adjustFormat( StackOfDocument[ Index ] )
            self.__determineEndOfDocument( EndOfDocuments, StackOfDocument[ Index ] )

        return ( "\n\n".join( StackOfDocument ), EndOfDocuments )

    def __determineEndOfDocument( self, EndOfDocuments: list, Document: str ):
        EndOfDocuments.append(  len( Document ) + EndOfDocuments[-1] + 2  )

    def __adjustFormat( self, Document: str ) -> str:
        return ComplexNormalizer.__Pattern.sub( r'\n', Document ).strip()

    def __filterSentencesAndSplitDocuments(
        self,
        ParsedBatchDocuments: Document,
        EndOfDocuments: list,
        Flags: str
    ) -> list:
        ParsedDocuments = list()
        ParsedDocument = list()

        for Sentence in ParsedBatchDocuments.sentences:
            Pos = self.__filter( ParsedDocument, Sentence.words, Flags )
            if Pos in EndOfDocuments:
                ParsedDocuments.append( self._reassemble( ParsedDocument ) )
                ParsedDocument = list()

        return ParsedDocuments

    def __filter( self, Document: list, Words: list, Flags: str ) -> int:
        for Word in Words:
            if "n" in Flags:
                self.__appendNouns( Document, Word )
            if "v" in Flags:
                self.__appendVerbs( Document, Word )
            if "a" in Flags:
                self.__appendAdj( Document, Word )
            if "y" in Flags:
                self.__appendSym( Document, Word )
            if "u" in Flags:
                self.__appendNumerals( Document, Word )

        return int( Words[ -1 ].misc.split( "|end_char=" )[ 1 ] )

    def __appendNouns( self, Filtered: list, Word: NLPToken ):
        self.__checkAndAppend( "NOUN", Filtered, Word )
        self.__checkAndAppend( "PROPN", Filtered, Word )

    def __appendVerbs( self, Filtered: list, Word: NLPToken ):
        self.__checkAndAppend( "VERB", Filtered, Word )

    def __appendAdj( self, Filtered: list, Word: NLPToken ):
        self.__checkAndAppend( "ADJ", Filtered, Word )

    def __appendSym( self, Filtered: list, Word: NLPToken ):
        self.__checkAndAppend( "SYM", Filtered, Word )

    def __appendNumerals( self, Filtered: list, Word: NLPToken ):
        self.__checkAndAppend( "NUM", Filtered, Word )

    def __checkAndAppend( self, Type, Filtered, Word: NLPToken ):
        if Word.upos == Type:
            self.__append( Filtered, Word )

    def __append( self, Filtered: list, Word: NLPToken ):
        Filtered.append( Word.lemma )

    class Factory( NormalizerFactory ):
        @staticmethod
        def getInstance() -> Normalizer:
            return ComplexNormalizer()
