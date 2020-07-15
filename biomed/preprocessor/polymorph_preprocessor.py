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
from multiprocessing import Process, Lock, Manager
from time import sleep
from math import ceil

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
        ComplexFlags: list,
        Lock: Lock
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
        self.__Lock = Lock

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
        PmIds, Documents = self.__cleanUpData(
            list( frame[ "pmid" ] ),
            list( frame[ "text" ] )
        )

        return self.__reflectOrExtract( PmIds, Documents, self.__toSortedString( flags ) )

    def __cleanUpData( self, PmIds: list, Documents: list ) -> tuple:
        Result = self.__FM.clean( PmIds, Documents )
        if not Result[ 0 ] or not Result[ 1 ]:
            raise RuntimeError( "ERROR: Empty Dataset detected." )

        return Result

    def __reflectOrExtract( self, PmIds: list, Document: list, Flags: str ) -> list:
        if not self.__isApplicable( Flags ):
            return Document
        else:
            return self.__runInParallelOrSequence( PmIds, Document, Flags )


    def __runInParallelOrSequence( self, PmIds: list, Document: list, Flags: str ) -> list:
        if not self.__ForkIt:
            return self.__extractDocument(
                PmIds,
                Document,
                Flags,
            )
        else:
            return self.__runInParallel(
                PmIds,
                Document,
                Flags
            )

    def __runInParallel( self, PmIds: list, Documents: list, Flags: str ) -> list:
        print( "Gathering already computed" )
        Ids, Documents = self.__filterAlreadyComputed( PmIds, Documents, Flags )

        print( "prepare documents" )
        self.__excuteRun(
            Ids,
            Documents,
            Flags
        )

        self.__saveOnDone()
        print( "Gathering output" )
        return self.__returnFromCache( PmIds, Flags )

    def __excuteRun( self, CacheIds: list, Documents: list, Flags: str ):
        if not CacheIds:
            return

        Jobs = list()
        BagOfCacheIds, BagOfDocuments = self.__splitInputs( CacheIds, Documents, Flags )

        self.__spawnJobs( Jobs, BagOfCacheIds, BagOfDocuments, Flags )
        sleep( 0 )# aka yield
        self.__waitUntilDone( Jobs )

    def __splitInputs( self, CacheIds: list, Documents: list, Flags: str ) -> tuple:
        SizeOfChunk = ceil( len( CacheIds ) / self.__Workers )
        SubsetOfIds = list( self.__computeChunk( SizeOfChunk, CacheIds ) )
        SubsetOfDocuments = list( self.__computeChunk( SizeOfChunk, Documents ) )

        return ( SubsetOfIds, SubsetOfDocuments )

    def __computeChunk( self, N: int, BagOfStuff: list ) -> list:
        for Index in range( 0, len( BagOfStuff ), N ):
            yield BagOfStuff[ Index:Index + N]

    def __spawnJobs( self, Jobs, BagOfCacheIds: list, BagOfDocuments: list, Flags: str ):
        # Note there could be more worker than stuff to process
        for Index in range( 0, len( BagOfCacheIds ) ):
            print( "Spawn job #{}". format( Index ) )
            Job = Process(
                target = PolymorphPreprocessor._run,
                args = (
                    self,
                    Index,
                    BagOfCacheIds[ Index ],
                    BagOfDocuments[ Index ],
                    Flags
                )
            )
            Jobs.append( Job )
            Job.start()

    def __waitUntilDone( self, Jobs ):
        for Job in Jobs:
            Job.join()

    @staticmethod
    def _run(
        This,
        Worker: int,
        CacheIds: list,
        Documents: list,
        Flags: str
    ):
        sleep(0)# aka yield

        This.__computeAndWriteToCache(
            CacheIds,
            Documents,
            Flags,
            Worker
        )

        print( "Job #{} is done!".format( Worker ) )

    def __extractDocument( self, PmIds: list, Documents: list, Flags: str ) -> list:
        print( "Gathering already computed" )
        Ids, Documents = self.__filterAlreadyComputed( PmIds, Documents, Flags )

        print( "Prepare documents" )
        self.__computeAndWriteToCache(
            Ids,
            Documents,
            Flags,
            0
        )

        self.__saveOnDone()
        print( "Gathering output" )
        return self.__returnFromCache( PmIds, Flags )

    def __saveOnDone( self ):
        if self.__SharedMemory.get( "Dirty" ):
            self.__SharedMemory.set( "Dirty", False )
            self.__save()

    def __returnFromCache( self, Ids: list, Flags: str ):
        ParsedDocuments = list()
        for Id in Ids:
            ParsedDocuments.append(
                self.__SharedMemory.get(
                    self.__createCacheKey( Id, Flags )
                )
            )

        return ParsedDocuments

    def __filterAlreadyComputed( self, PmIds: list, Documents: list,  Flags: str ) -> tuple:
        SubsetOfIds = list()
        SubsetOfDocuments = list()
        for Index in range( 0, len( PmIds ) ):
            CacheId = self.__createCacheKey( PmIds[ Index ], Flags )
            if not self.__SharedMemory.has( CacheId ):
                self.__SharedMemory.set( "Dirty", True )
                SubsetOfIds.append( CacheId )
                SubsetOfDocuments.append( Documents[ Index ] )

        return ( SubsetOfIds, SubsetOfDocuments )

    def __computeAndWriteToCache( self, CacheIds: list, Documents: list, Flags: str, Worker: int ):
        if not CacheIds:
            return

        self.__writeToCache(
            CacheIds,
            self.__normalize( Documents, Flags, Worker )
        )

    def __writeToCache( self, CacheIds: list, NormalizedDocuments: list ):
        self.__Lock.acquire()
        for Index in range( 0, len( CacheIds ) ):
            self.__SharedMemory.set(
                CacheIds[ Index ],
                NormalizedDocuments[ Index ]
            )
        self.__Lock.release()

    def __normalize( self, StackOfDocuments: list, Flags: str, Worker: int ) -> list:
        ParsedDocuments = StackOfDocuments

        if self.__useComplex( Flags ):
            ParsedDocuments = self.__Complex[ Worker ].apply( ParsedDocuments, Flags )

        if self.__useSimple( Flags ):
            ParsedDocuments = self.__Simple[ Worker ].apply( ParsedDocuments, Flags )

        return ParsedDocuments

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

    def __createCacheKey( self, PmId: int, Flags: str ) -> str:
        return "{}{}".format( PmId, Flags )

    def __toSortedString( self, Str: str ) -> str:
        Tmp = list( Str )
        Tmp.sort()
        return "".join( Tmp )

    def __save( self ):
        if self.__SharedMemory.size() > 0:
            self.__AlreadyProcessed.set( "hardId42", self.__SharedMemory.toDict() )

    class Factory( PreProcessorFactory ):
        __FacilityManager = MariosFacilityManager.Factory.getInstance()
        __Simple = SimpleNormalizer.Factory
        __SimpleFlags = [ "s", "l", "w" ]
        __Complex = ComplexNormalizer.Factory
        __ComplexFlags = [ "n", "v", "a", "u", "y" ]

        @staticmethod
        def getInstance( Properties: PropertiesManager ) -> PreProcessor:
            FileCache = NumpyArrayFileCache.Factory.getInstance(
                Properties.cache_dir
            )

            return PolymorphPreprocessor(
                PolymorphPreprocessor.Factory.__FacilityManager,
                Properties.preprocessing[ "workers" ],
                FileCache,
                PolymorphPreprocessor.Factory.__loadSharedMemory( FileCache ),
                PolymorphPreprocessor.Factory.__Simple,
                PolymorphPreprocessor.Factory.__SimpleFlags,
                PolymorphPreprocessor.Factory.__Complex,
                PolymorphPreprocessor.Factory.__ComplexFlags,
                Manager().Lock(),
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
