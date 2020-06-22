import os as OS
from biomed.preprocessor.cache.cache import Cache
from biomed.preprocessor.cache.cache import FileCacheFactory
from multiprocessing import Manager, Lock
import numpy

class NumpyArrayFileCache( Cache ):
    def __init__( self, CacheDir: str, Lock: Lock ):
        self.__CacheDir = CacheDir
        self.__Lock = Lock

    def __keyToFileName( self, Key: str ) -> str:
        return OS.path.join( self.__CacheDir, "{}.npy".format( Key ) )

    def has( self, Key: str ) -> bool:
        File = self.__keyToFileName( Key )
        return self.__has( File )

    def __has( self, File: str ) -> bool:
        return OS.path.exists( File ) and OS.path.isfile( File )

    def get( self, Key: str ):
        File = self.__keyToFileName( Key )
        if self.__has( File ):
            return numpy.load( File, allow_pickle=True ).item()
        else:
            return None

    def set( self, Key: str, Value ):
        File = self.__keyToFileName( Key )
        self.__Lock.acquire()
        numpy.save( File, Value )
        self.__Lock.release()

    def size( self ):
        raise NotImplementedError()

    def toDict( self ):
        raise NotImplementedError()

    class Factory( FileCacheFactory ):
        __Manager = Manager()

        @staticmethod
        def __checkDir( Dir, Readable=True, Writeable=True ):
            if OS.path.isdir( Dir ) is False:
                raise RuntimeError( "{} not found.".format( Dir ) )

            return NumpyArrayFileCache.Factory.__checkAccess(
                Dir,
                Readable,
                Writeable
            )

        @staticmethod
        def __checkAccess( Path, Readable, Writeable ):
            if True is Readable and OS.access( Path, OS.R_OK ) is False:
                raise RuntimeError( "{} is not readable".format( Path ) )

            if True is Writeable and OS.access( Path, OS.W_OK ) is False:
                raise RuntimeError( "{} is not writeable".format( Path ) )

        @staticmethod
        def __toAbsPath( Path: str ) -> str:
            if OS.path.isabs( Path ):
                return OS.path.abspath( Path )
            else:
                return Path

        @staticmethod
        def getInstance( PathToCacheDir: str ) -> Cache:
            NumpyArrayFileCache.Factory.__checkDir(
                NumpyArrayFileCache.Factory.__toAbsPath( PathToCacheDir )
            )

            return NumpyArrayFileCache(
                PathToCacheDir,
                NumpyArrayFileCache.Factory.__Manager.Lock()
            )
