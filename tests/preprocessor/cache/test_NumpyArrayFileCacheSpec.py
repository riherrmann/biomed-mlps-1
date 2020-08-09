import os as OS
import unittest
from unittest.mock import patch, MagicMock
import numpy
from biomed.preprocessor.cache.cache import Cache
from biomed.preprocessor.cache.numpyArrayFileCache import NumpyArrayFileCache
from biomed.properties_manager import PropertiesManager

class NumpyArrayFileCacheSpec( unittest.TestCase ):
    __Path = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), 'testTmp' ) )

    def setUp( self ):
        OS.mkdir( self.__Path, 0o777 )
        self.__Files = list()
        self.__checkDirM = patch( 'biomed.preprocessor.cache.numpyArrayFileCache.checkDir' )
        self.__checkDir = self.__checkDirM.start()

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
        self.__checkDirM.stop()

    def __fakeLocator( self, _, __ ):
        PM = PropertiesManager()
        PM.cache_dir = NumpyArrayFileCacheSpec.__Path

        return PM

    def test_it_checks_a_the_dir_on_init( self ):
        NumpyArrayFileCache.Factory.getInstance( self.__fakeLocator )
        self.__checkDir.assert_called_once()

    def test_it_is_a_Cache( self ):
        MyCache = NumpyArrayFileCache.Factory.getInstance( self.__fakeLocator )
        self.assertTrue( isinstance( MyCache, Cache ) )

    def test_it_depends_on_the_properties_mananger( self ):
        def fakeLocator( ServiceKey: str, ExpectedType ):
            if ServiceKey != "properties" or ExpectedType != PropertiesManager:
                raise RuntimeError( "Unexpected Service" )
            return PropertiesManager()

        ServiceGetter = MagicMock()
        ServiceGetter.side_effect = fakeLocator

        NumpyArrayFileCache.Factory.getInstance( ServiceGetter )
        ServiceGetter.assert_called_once()

    def test_it_tells_if_contains_a_id( self ):
        self.__createTestFile( [ 1, 2, 3 ], "1.npy" )

        MyCache = NumpyArrayFileCache.Factory.getInstance( self.__fakeLocator )
        self.assertTrue( MyCache.has( "1" ) )
        self.assertFalse( MyCache.has( "2" ) )

    def test_it_returns_a_stored_value( self ):
        Stored = { "a": [ 1, 2, 3 ] }
        self.__createTestFile( Stored, "1.npy" )

        MyCache = NumpyArrayFileCache.Factory.getInstance( self.__fakeLocator )
        self.assertDictEqual(
            Stored,
            MyCache.get( "1" )
        )

    def test_it_returns_none_if_the_key_does_not_exists( self ):
        MyCache = NumpyArrayFileCache.Factory.getInstance( self.__fakeLocator )
        self.assertEqual(
            None,
            MyCache.get( "23" )
        )

    def test_it_stores_given_data( self ):
        ToStore = [ 1, 2, 3 ]

        MyCache = NumpyArrayFileCache.Factory.getInstance( self.__fakeLocator )

        MyCache.set( "1", ToStore )
        self.assertListEqual(
            ToStore,
            list( numpy.load( OS.path.join( self.__Path, "1.npy"  ) ) )
        )

        self.__Files.append( OS.path.join( self.__Path, "1.npy"  ) )

    def test_it_overwrites_stored_data( self ):
        ToStore = [ 1, 2, 3 ]
        self.__createTestFile( ToStore, "1.npy" )

        ToStore = [ 4, 5, 6 ]
        MyCache = NumpyArrayFileCache.Factory.getInstance( self.__fakeLocator )
        MyCache.set( "1", ToStore )
        self.assertListEqual(
            ToStore,
            list( numpy.load( OS.path.join( self.__Path, "1.npy"  ) ) )
        )

    def __del__( self ):
        self.__remove()
