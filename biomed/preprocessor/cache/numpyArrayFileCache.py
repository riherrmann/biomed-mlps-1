import os as OS
from biomed.preprocessor.cache.cache import Cache
from biomed.preprocessor.cache.cache import CacheFactory
import biomed.services as Services
from biomed.properties_manager import PropertiesManager
from biomed.utils.dir_checker import checkDir, toAbsPath
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

    class Factory( CacheFactory ):
        @staticmethod
        def getInstance() -> Cache:
            PathToCacheDir = Services.getService( "properties", PropertiesManager ).cache_dir

            checkDir( toAbsPath( PathToCacheDir ) )

            return NumpyArrayFileCache(
                PathToCacheDir,
                Manager().Lock()
            )
