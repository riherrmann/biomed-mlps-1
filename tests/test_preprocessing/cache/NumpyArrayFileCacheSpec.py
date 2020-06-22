import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
import numpy
from biomed.preprocessor.cache.cache import Cache
from biomed.preprocessor.cache.numpyArrayFileCache import NumpyArrayFileCache

class NumpyArrayFileCacheSpec( unittest.TestCase ):
    def setUp( self ):
        self.__Path = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), 'testTmp' ) )
        OS.mkdir( self.__Path, 0o777 )
        self.__Files = list()

    def __remove( self ):
        if self.__Files:
            for File in self.__Files:
                OS.remove( File )

            self.__Files = None

        if self.__Path:
            OS.rmdir( self.__Path )
            self.__Path = None

    # a bit stupid
    def __createTestFile( self, Content, FileName ):
        numpy.save( OS.path.join( self.__Path, FileName ), Content )
        self.__Files.append( OS.path.join( self.__Path, FileName ) )

    def tearDown( self ):
        self.__remove()

    def it_fails_if_the_cache_dir_does_not_exists( self ):
        self.__remove()
        self.assertRaises( RuntimeError )

    def it_fails_if_the_cache_dir_is_not_readable( self ):
        OS.chmod( self.__Path, 0o100 )
        self.assertRaises( RuntimeError )

    def it_fails_if_the_cache_dir_is_not_writeable( self ):
        OS.chmod( self.__Path, 0o660 )
        self.assertRaises( RuntimeError )

    def it_is_a_Cache( self ):
        MyCache = NumpyArrayFileCache.Factory.getInstance( self.__Path )
        self.assertTrue( isinstance( MyCache, Cache ) )

    def it_tells_if_contains_a_id( self ):
        self.__createTestFile( [ 1, 2, 3 ], "1.npy" )
        MyCache = NumpyArrayFileCache.Factory.getInstance( self.__Path )
        self.assertTrue( MyCache.has( "1" ) )
        self.assertFalse( MyCache.has( "2" ) )

    def it_returns_a_stored_value( self ):
        Stored = [ 1, 2, 3 ]
        self.__createTestFile( Stored, "1.npy" )
        MyCache = NumpyArrayFileCache.Factory.getInstance( self.__Path )
        self.assertListEqual(
            Stored,
            list( MyCache.get( "1" ) )
        )

    def it_returns_none_if_the_key_does_not_exists( self ):
        MyCache = NumpyArrayFileCache.Factory.getInstance( self.__Path )
        self.assertEqual(
            None,
            MyCache.get( "23" )
        )

    def it_stores_given_data( self ):
        ToStore = [ 1, 2, 3 ]
        MyCache = NumpyArrayFileCache.Factory.getInstance( self.__Path )
        MyCache.set( "1", ToStore )
        self.assertListEqual(
            ToStore,
            list( numpy.load( OS.path.join( self.__Path, "1.npy"  ) ) )
        )

        self.__Files.append( OS.path.join( self.__Path, "1.npy"  ) )

    def it_overwrites_stored_data( self ):
        ToStore = [ 1, 2, 3 ]
        self.__createTestFile( ToStore, "1.npy" )

        ToStore = [ 4, 5, 6 ]
        MyCache = NumpyArrayFileCache.Factory.getInstance( self.__Path )
        MyCache.set( "1", ToStore )
        self.assertListEqual(
            ToStore,
            list( numpy.load( OS.path.join( self.__Path, "1.npy"  ) ) )
        )

    def __del__( self ):
        self.__remove()
