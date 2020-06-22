from biomed.preprocessor.pre_processor import PreProcessor
from biomed.preprocessor.pre_processor import PreProcessorFactory
from biomed.preprocessor.normalizer.normalizer import NormalizerFactory
from biomed.preprocessor.normalizer.simpleNormalizer import SimpleNormalizer
from biomed.preprocessor.normalizer.complexNormalizer import ComplexNormalizer
from biomed.preprocessor.cache.cache import Cache
from biomed.preprocessor.cache.sharedMemoryCache import SharedMemoryCache
from biomed.preprocessor.cache.numpyArrayFileCache import NumpyArrayFileCache
from biomed.preprocessor.facilitymanager.facility_manager import FacilityManager
from biomed.preprocessor.facilitymanager.mFacilityManager import MariosFacilityManager
from biomed.properties_manager import PropertiesManager
from pandas import DataFrame
from nltk import sent_tokenize
from multiprocessing import Process
from time import sleep

class PolymorphPreprocessor( PreProcessor ):
    def __init__(
        self,
        FM: FacilityManager,
        Workers: int,
        AlreadyProcessed: Cache,
        Shared: Cache,
        Simple: NormalizerFactory,
        SimpleFlags: list,
        Complex: NormalizerFactory,
        ComplexFlags: list
    ):
        self.__FM = FM
        self.__AlreadyProcessed = AlreadyProcessed
        self.__SharedMemory = Shared
        Workers = max( Workers, 1 )
        self.__ForkIt = False if Workers == 1 else True
        self.__Workers = Workers
        self.__Cache = Cache
        self.__SimpleFlags = SimpleFlags
        self.__ComplexFlags = ComplexFlags
        self.__prepareNormalizers( Simple, Complex, Workers )

    def __prepareNormalizers(
        self,
        SimpleFactory: NormalizerFactory,
        ComplexFactory: NormalizerFactory,
        Amount: int
    ):
        self.__Simple = list()
        self.__Complex = list()
        for Index in range( 0, Amount ):
            self.__Simple.append( SimpleFactory.getInstance() )
            self.__Complex.append( ComplexFactory.getInstance() )

    def preprocess_text_corpus( self, frame: DataFrame, flags: str ) -> list:
        self.__SharedMemory.set( "Dirty", False )
        PmIds, Texts = self.__cleanUpData(
            list( frame[ "pmid" ] ),
            list( frame[ "text" ] )
        )

        return self.__reflectOrExtract( PmIds, Texts, flags )

    def __cleanUpData( self, PmIds: list, Texts: list ) -> tuple:
        Result = self.__FM.clean( PmIds, Texts )
        if not Result[ 0 ] or not Result[ 1 ]:
            raise RuntimeError( "ERROR: Empty Dataset detected." )

        return Result

    def __reflectOrExtract( self, PmIds: list, Text: list, Flags: str ) -> list:
        if not self.__isApplicable( Flags ):
            return Text
        else:
            return self.__runInParallelOrSequence( PmIds, Text, Flags )


    def __runInParallelOrSequence( self, PmIds: list, Text: list, Flags: str ) -> list:
        if not self.__ForkIt:
            return self.__extractText(
                PmIds,
                Text,
                Flags,
                0
            )
        else:
            return self.__runInParallel(
                PmIds,
                Text,
                Flags
            )

    def __runInParallel( self, PmIds: list, Text: list, Flags: str ) -> list:
        Jobs = list()
        PairedValues = self.__splitInputs( PmIds, Text, Flags )
        self.__spawnJobs( Jobs, PairedValues, Flags )
        sleep( 0 )# aka yield
        self.__waitUntilDone( Jobs )

        return self.__returnParallelResults( PmIds, Flags )

    def __splitInputs( self, PmIds: list, Text: list, Flags: str ) -> list:
        Buckets = list()
        for Index in range( 0, self.__Workers ):
            Buckets.append( list() )

        for Index in range( 0, len( PmIds ) ):
            Buckets[ Index % self.__Workers ].append(
                (
                    self.__createCacheKey( PmIds[ Index ], Flags ),
                    Text[ Index ]
                )
            )

        return Buckets

    def __spawnJobs( self, Jobs, PairedValues: list, Flags: str ):
        for Index in range( 0, self.__Workers ):
            Job = Process(
                target = PolymorphPreprocessor._run,
                args = ( self, Index, PairedValues[ Index ], Flags )
            )
            Jobs.append( Job )
            Job.start()

    def __waitUntilDone( self, Jobs ):
        for Job in Jobs:
            Job.join()

    def __returnParallelResults( self, PmIds: list, Flags: str ) -> list:
        print( "Gathering computed sets" )
        Results = list()
        for PmId in PmIds:
            Results.append(
                self.__SharedMemory.get(
                    self.__createCacheKey( PmId, Flags )
                )
            )

        self.__saveOnDone()
        return Results

    @staticmethod
    def _run(
        This,
        Worker: int,
        PairedValues: list,
        Flags: str
    ):
        sleep(0)# aka yield
        for Value in PairedValues:
            This.__multiExtractText( Value, Flags, Worker )
        print( "Job is Done!" )

    def __multiExtractText( self, Paired: tuple, Flags: str, Worker: int ):
        print( 'Preprocess {} in worker {}'.format( Paired[ 0 ], Worker ) )

        if not self.__SharedMemory.has( Paired[ 0 ] ):
            self.__applyTextNormalizerAndCache( Paired[ 0 ], Paired[ 1 ], Flags, Worker )
            self.__SharedMemory.set(  "Dirty", True )

    def __extractText( self, PmIds: list, Text: list, Flags: str, Worker: int ) -> list:
        for Index in range( 0, len( Text ) ):
            Text[ Index ] = self.__useCacheOrNormalizer(
                PmIds[ Index ],
                Text[ Index ],
                Flags,
                Worker,
            )

        self.__saveOnDone()
        return Text

    def __saveOnDone( self ):
        if self.__SharedMemory.get( "Dirty" ):
            self.__SharedMemory.set( "Dirty", False )
            self.__save()

    def __useCacheOrNormalizer(
        self,
        PmId: int,
        Text: str,
        Flags: str,
        Worker: int
    ) -> str:
        CacheKey = self.__createCacheKey( PmId, Flags )
        print( 'Preprocess {}'.format( CacheKey ) )

        if self.__SharedMemory.has( CacheKey ):
            return self.__SharedMemory.get( CacheKey )
        else:
            self.__SharedMemory.set( "Dirty", True )
            return self.__applyTextNormalizerAndCache( CacheKey, Text, Flags, Worker )

    def __createCacheKey( self, PmId: int, Flags: str ) -> str:
        Flags = self.__toSortedString( Flags )
        return "{}{}".format( PmId, Flags )

    def __toSortedString( self, Str: str ) -> str:
        Tmp = list( Str )
        Tmp.sort()
        return "".join( Tmp )

    def __applyTextNormalizerAndCache(
        self,
        CacheKey: str,
        Text: str,
        Flags: str,
        Worker: int
    ) -> str:
        Result = self.__applyTextNormalizer( Text, Flags, Worker )
        self.__SharedMemory.set( CacheKey, Result )
        return Result

    def __applyTextNormalizer( self, Text: str, Flags: str, Worker: int ) -> str:
        return self.__reassemble(
            self.__normalize( sent_tokenize( Text ), Flags, Worker )
        )

    def __normalize( self, Sentences: list, Flags: str, Worker: int ) -> list:
        ParsedSentences = list()
        for Sentence in Sentences:
            ParsedSentences.append(
                self.__normalizePerSentence( Sentence, Flags, Worker )
            )

        return ParsedSentences

    def __normalizePerSentence( self, Text: str, Flags: str, Worker: int ) -> str:
        ParsedSentence = Text

        if self.__useComplex( Flags ):
            ParsedSentence = self.__Complex[ Worker ].apply( ParsedSentence, Flags )

        if self.__useSimple( Flags ):
            ParsedSentence = self.__Simple[ Worker ].apply( ParsedSentence, Flags )

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

    def __save( self ):
        if self.__SharedMemory.size() > 0:
            self.__AlreadyProcessed.set( "hardId42", self.__SharedMemory.toDict() )

    class Factory( PreProcessorFactory ):
        __FacilityManager = MariosFacilityManager.Factory.getInstance()
        __Simple = SimpleNormalizer.Factory
        __SimpleFlags = [ "s", "l", "w" ]
        __Complex = ComplexNormalizer.Factory
        __ComplexFlags = [ "n", "v", "a" ]

        @staticmethod
        def getInstance( Properties: PropertiesManager ) -> PreProcessor:
            FileCache = NumpyArrayFileCache.Factory.getInstance(
                Properties.cache_dir
            )

            return PolymorphPreprocessor(
                PolymorphPreprocessor.Factory.__FacilityManager,
                Properties.workers,
                FileCache,
                PolymorphPreprocessor.Factory.__loadSharedMemory( FileCache ),
                PolymorphPreprocessor.Factory.__Simple,
                PolymorphPreprocessor.Factory.__SimpleFlags,
                PolymorphPreprocessor.Factory.__Complex,
                PolymorphPreprocessor.Factory.__ComplexFlags
            )

        @staticmethod
        def __loadSharedMemory( FileCache: Cache ) -> Cache:
            SharedMemory = SharedMemoryCache.Factory.getInstance()

            if FileCache.has( "hardId42" ):
                PolymorphPreprocessor.Factory.__loadIntoSharedMemory( FileCache, SharedMemory )

            return SharedMemory

        @staticmethod
        def __loadIntoSharedMemory( FileCache: Cache, SharedMemory: Cache ):
            StaticValues = FileCache.get( "hardId42" )
            for Key in StaticValues:
                SharedMemory.set( Key, StaticValues[ Key ] )
