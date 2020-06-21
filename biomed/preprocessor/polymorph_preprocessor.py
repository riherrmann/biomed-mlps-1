from biomed.preprocessor.pre_processor import PreProcessor
from biomed.preprocessor.pre_processor import PreProcessorFactory
from biomed.preprocessor.normalizer.normalizer import Normalizer
from biomed.preprocessor.normalizer.simpleNormalizer import SimpleNormalizer
from biomed.preprocessor.normalizer.complexNormalizer import ComplexNormalizer
from biomed.preprocessor.cache.cache import Cache
from biomed.preprocessor.cache.sharedMemoryCache import SharedMemoryCache
from biomed.preprocessor.cache.numpyArrayFileCache import NumpyArrayFileCache
from biomed.properties_manager import PropertiesManager
from pandas import DataFrame
from nltk import sent_tokenize
from hashlib import md5
from multiprocessing import Pool


class PolymorphPreprocessor( PreProcessor ):
    def __init__(
        self,
        Workers,
        AlreadyProcessed: Cache,
        Shared: Cache,
        Simple: Normalizer,
        SimpleFlags: list,
        Complex: Normalizer,
        ComplexFlags: list
    ):
        self.__AlreadyProcessed = AlreadyProcessed
        self.__SharedMemory = Shared
        Workers = max( Workers, 1 )
        self.__ForkIt = False if Workers == 1 else True
        self.__Workers = Workers
        self.__Cache = Cache
        self.__SimpleFlags = SimpleFlags
        self.__Simple = Simple
        self.__ComplexFlags = ComplexFlags
        self.__Complex = Complex

    def preprocess_text_corpus( self, frame: DataFrame, flags: str ) -> list:
        return self.__reflectOrExtract(
            list( frame[ "pmid" ] ),
            list( frame[ "text" ] ),
            flags
        )

    def __reflectOrExtract( self, Pmid: list, Text: list, Flags: str ) -> list:
        if not self.__isApplicable( Flags ):
            return Text
        else:
            return self.__getCachedListOrCompute( Pmid, Text, Flags )

    def __getCachedListOrCompute( self, Pmid: list, Text: list, Flags: str ) -> list:
        SetId = self.__computeSetId( Pmid, Flags )
        if self.__AlreadyProcessed.has( SetId ):
            return self.__AlreadyProcessed.get( SetId )
        else:
            return self.__cacheAndReturn(
                SetId,
                self.__runInParalellOrSequence( Pmid, Text, Flags )
            )

    def __computeSetId( self, Pmid: list, Flags: str ) -> str:
        Pmid = list( Pmid ) # note we have to copy it regarding to not destroy the order
        Pmid.sort()
        Pmid = [ str(integer) for integer in Pmid ]
        Flags = self.__toSortedString( Flags )
        SetId = md5()
        SetId.update( "-".join( Pmid ).encode( 'utf-8' ) )
        SetId.update( "-{}".format( Flags ).encode( 'utf-8' ) )
        return str( SetId.hexdigest() )

    def __cacheAndReturn( self, SetId: str, ProcessedText: list ) -> list:
        self.__AlreadyProcessed.set( SetId, ProcessedText )
        return ProcessedText

    def __extractText( self, Pmid: list, Text: list, Flags: str ) -> list:
        for Index in range( 0, len( Text ) ):
            print( 'Preprocess {}'.format( Pmid[ Index ] ) )
            Text[ Index ] = self.__useCacheOrNormalizer(
                Pmid[ Index ],
                Text[ Index ],
                Flags
            )

        return Text

    def __runInParalellOrSequence( self, Pmid: list, Text: list, Flags: str ) -> list:
        if not self.__ForkIt:
            return self.__extractText(
                Pmid,
                Text,
                Flags
            )
        else:
            return self.__runInParalell(
                Pmid,
                Text,
                Flags
            )

    def __runInParalell( self, Pmid: list, Text: list, Flags: str ) -> list:
        with Pool( self.__Workers ) as Runner:
            Runner.map(
                PolymorphPreprocessor._run,
                self.__splitInputs( Pmid, Text, Flags )
            )

        return self.__extractText(
            Pmid,
            Text,
            Flags
        )

    @staticmethod
    def _run( Paired: tuple ):
        This = Paired[ 3 ]
        This.__multiExtractText( Paired )
        return ""

    def __multiExtractText( self, Paired: tuple ):
        CacheKey = self.__createCacheKey( Paired[ 0 ], Paired[ 2 ] )
        print( 'Preprocess {}'.format( Paired[ 0 ] ) )

        if not self.__Cache.has( CacheKey ):
            self.__applyTextNormalizerAndCache( CacheKey, Paired[ 1 ], Paired[ 2 ] )

    def __splitInputs( self, Pmid: list, Text: list, Flags: str ) -> list:
        PairedValues = list()

        for Index in range( 0, len( Pmid ) ):
            PairedValues.append( ( Pmid[ Index ], Text[ Index ], Flags, self ) )

        return PairedValues

    def __useCacheOrNormalizer( self, Pmid: int, Text: str, Flags: str ) -> str:
        CacheKey = self.__createCacheKey( Pmid, Flags )
        if self.__SharedMemory.has( CacheKey ):
            return self.__SharedMemory.get( CacheKey )
        else:
            return self.__applyTextNormalizerAndCache( CacheKey, Text, Flags )

    def __createCacheKey( self, Pmid: int, Flags: str ) -> str:
        Flags = self.__toSortedString( Flags )
        return "{}{}".format( Pmid, Flags )

    def __toSortedString( self, Str: str ) -> str:
        Tmp = list( Str )
        Tmp.sort()
        return "".join( Tmp )

    def __applyTextNormalizerAndCache( self, CacheKey: str, Text: str, Flags: str ) -> str:
        Result = self.__applyTextNormalizer( Text, Flags )
        self.__SharedMemory.set( CacheKey, Result )
        return Result

    def __applyTextNormalizer( self, Text: str, Flags: str ) -> str:
        return self.__reassemble(
            self.__normalize( sent_tokenize( Text ), Flags )
        )

    def __normalize( self, Sentences: list, Flags: str ) -> list:
        ParsedSentences = list()
        for Sentence in Sentences:
            ParsedSentences.append( self.__normalizePerSentence( Sentence, Flags ) )

        return ParsedSentences

    def __normalizePerSentence( self, Text: str, Flags: str ) -> str:
        ParsedSentence = Text
        if self.__useComplex( Flags ):
            ParsedSentence = self.__Complex.apply( ParsedSentence, Flags )

        if self.__useSimple( Flags ):
            ParsedSentence = self.__Simple.apply( ParsedSentence, Flags )

        return ParsedSentence

    def __isApplicable( self, Flags: str ) -> bool:
        return self.__useSimple( Flags ) or self.__useComplex( Flags )

    def __useSimple( self, Flags: str ) -> bool:
        for Flag in Flags:
            if Flag in self.__SimpleFlags:
                return True
        else:
            return False

    def __useComplex( self, Flags: str ) -> bool:
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
        __DistrubutedCache = SharedMemoryCache.Factory.getInstance()

        @staticmethod
        def getInstance( Properties: PropertiesManager ) -> PreProcessor:
            return PolymorphPreprocessor(
                Properties.workers,
                NumpyArrayFileCache.Factory.getInstance(
                    Properties.cache_dir
                ),
                PolymorphPreprocessor.Factory.__DistrubutedCache,
                PolymorphPreprocessor.Factory.__Simple,
                PolymorphPreprocessor.Factory.__SimpleFlags,
                PolymorphPreprocessor.Factory.__Complex,
                PolymorphPreprocessor.Factory.__ComplexFlags
            )
