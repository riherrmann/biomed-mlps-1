from biomed.preprocessor.preprocessor import Preprocessor
from biomed.preprocessor.preprocessor import PreprocessorFactory
from biomed.preprocessor.normalizer.normalizer import NormalizerFactory
from biomed.preprocessor.cache.cache import Cache
from biomed.properties_manager import PropertiesManager
from biomed.services_getter import ServiceGetter
from pandas import Series
from multiprocessing import Process, Lock, Manager
from time import sleep
from math import ceil

class PolymorphPreprocessor( Preprocessor ):
    def __init__(
        self,
        PM: PropertiesManager,
        AlreadyProcessed: Cache,
        Shared: Cache,
        Simple: NormalizerFactory,
        Complex: NormalizerFactory,
        Lock: Lock
    ):
        self.__Properties = PM
        self.__AlreadyProcessed = AlreadyProcessed
        self.__SharedMemory = Shared
        self.__Cache = Cache
        self.__SimpleFactory = Simple
        self.__ComplexFactory = Complex
        self.__SimpleFlags = Simple.getApplicableFlags()
        self.__ComplexFlags = Complex.getApplicableFlags()
        self.__Lock = Lock

    def preprocessCorpus( self, Ids: Series, Corpus: Series ) -> Series:
        self.__SharedMemory.set( "Dirty", False )
        Ids = list( Ids )
        if not Ids:
            raise RuntimeError( "No processable data had been given" )

        return Series(
            self.__reflectOrExtract(
                Ids,
                list( Corpus ),
                self.__toSortedString( self.__Properties.preprocessing[ 'variant' ] ),
                self.__Properties.preprocessing[ 'workers' ]
            ),
            index = Ids
        )

    def __reflectOrExtract(
        self,
        PmIds: list,
        Document: list,
        Flags: str,
        Workers: int
    ) -> list:
        if not self.__isApplicable( Flags ):
            return Document
        else:
            return self.__runInParallelOrSequence( PmIds, Document, Flags, Workers )

    def __isParalell( self, Workers: int ) -> bool:
        return Workers > 1

    def __runInParallelOrSequence(
        self,
        PmIds: list,
        Document: list,
        Flags: str,
        Workers: int
    ) -> list:
        if not self.__isParalell( Workers ):
            return self.__extractDocument(
                PmIds,
                Document,
                Flags,
            )
        else:
            return self.__runInParallel(
                PmIds,
                Document,
                Flags,
                Workers
            )

    def __runInParallel(
        self,
        PmIds: list,
        Documents: list,
        Flags: str,
        Workers: int
    ) -> list:
        print( "Gathering already computed" )
        Ids, Documents = self.__filterAlreadyComputed( PmIds, Documents, Flags )

        print( "prepare documents" )
        self.__excuteRun(
            Ids,
            Documents,
            Flags,
            Workers
        )

        self.__saveOnDone()
        print( "Gathering output" )
        return self.__returnFromCache( PmIds, Flags )

    def __excuteRun( self, CacheIds: list, Documents: list, Flags: str, Workers ):
        if not CacheIds:
            return

        Jobs = list()
        BagOfCacheIds, BagOfDocuments = self.__splitInputs( CacheIds, Documents, Flags, Workers )
        self.__prepareNormalizers( Workers )

        self.__spawnJobs( Jobs, BagOfCacheIds, BagOfDocuments, Flags )
        sleep( 0 )# aka yield
        self.__waitUntilDone( Jobs )

    def __prepareNormalizers( self, Amount: int ):
        self.__Simple = list()
        self.__Complex = list()
        for Index in range( 0, Amount ):
            self.__Simple.append( self.__SimpleFactory.getInstance() )
            self.__Complex.append( self.__ComplexFactory.getInstance() )

    def __splitInputs( self, CacheIds: list, Documents: list, Flags: str, Workers ) -> tuple:
        SizeOfChunk = ceil( len( CacheIds ) / Workers )
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
        self.__prepareNormalizers( 1 )

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

    class Factory( PreprocessorFactory ):
        @staticmethod
        def getInstance( getService: ServiceGetter ) -> Preprocessor:
            FileCache = getService( "preprocessor.cache.persistent", Cache )

            return PolymorphPreprocessor(
                getService( "properties", PropertiesManager ),
                FileCache,
                PolymorphPreprocessor.Factory.__loadSharedMemory( getService, FileCache ),
                getService( "preprocessor.normalizer.simple", NormalizerFactory ),
                getService( "preprocessor.normalizer.complex", NormalizerFactory ),
                Manager().Lock(),
            )

        @staticmethod
        def __loadSharedMemory( getService: ServiceGetter, FileCache: Cache ) -> Cache:
            SharedMemory = getService( "preprocessor.cache.shared", Cache )

            if FileCache.has( "hardId42" ):
                PolymorphPreprocessor.Factory.__loadIntoSharedMemory( FileCache, SharedMemory )

            return SharedMemory

        @staticmethod
        def __loadIntoSharedMemory( FileCache: Cache, SharedMemory: Cache ):
            StaticValues = FileCache.get( "hardId42" )
            for Key in StaticValues:
                SharedMemory.set( Key, StaticValues[ Key ] )
