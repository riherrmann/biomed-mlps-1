import os as OS
import unittest
from unittest.mock import MagicMock, patch
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

    def fakeLocator( self, _, __ ):
        PM = PropertiesManager()
        PM.cache_dir = NumpyArrayFileCacheSpec.__Path

        return PM

    @patch( 'biomed.preprocessor.cache.numpyArrayFileCache.Services.getService' )
    def test_it_checks_a_the_dir_on_init( self, ServiceGetter: MagicMock ):
        ServiceGetter.side_effect = self.fakeLocator

        NumpyArrayFileCache.Factory.getInstance()
        self.__checkDir.assert_called_once()

    @patch( 'biomed.preprocessor.cache.numpyArrayFileCache.Services.getService' )
    def test_it_is_a_Cache( self, ServiceGetter: MagicMock ):
        ServiceGetter.side_effect = self.fakeLocator

        MyCache = NumpyArrayFileCache.Factory.getInstance()
        self.assertTrue( isinstance( MyCache, Cache ) )

    @patch( 'biomed.preprocessor.cache.numpyArrayFileCache.Services.getService' )
    def test_it_depends_on_the_properties_mananger( self, ServiceGetter: MagicMock ):
        def fakeLocator( ServiceKey: str, ExpectedType ):
            if ServiceKey != "properties" or ExpectedType != PropertiesManager:
                raise RuntimeError( "Unexpected Service" )
            return PropertiesManager()

        ServiceGetter.side_effect = fakeLocator

        NumpyArrayFileCache.Factory.getInstance()

    @patch( 'biomed.preprocessor.cache.numpyArrayFileCache.Services.getService' )
    def test_it_tells_if_contains_a_id( self, ServiceGetter: MagicMock ):
        self.__createTestFile( [ 1, 2, 3 ], "1.npy" )

        ServiceGetter.side_effect = self.fakeLocator

        MyCache = NumpyArrayFileCache.Factory.getInstance()
        self.assertTrue( MyCache.has( "1" ) )
        self.assertFalse( MyCache.has( "2" ) )

    @patch( 'biomed.preprocessor.cache.numpyArrayFileCache.Services.getService' )
    def test_it_returns_a_stored_value( self, ServiceGetter: MagicMock ):
        Stored = { "a": [ 1, 2, 3 ] }
        self.__createTestFile( Stored, "1.npy" )

        ServiceGetter.side_effect = self.fakeLocator

        MyCache = NumpyArrayFileCache.Factory.getInstance()
        self.assertDictEqual(
            Stored,
            MyCache.get( "1" )
        )

    @patch( 'biomed.preprocessor.cache.numpyArrayFileCache.Services.getService' )
    def test_it_returns_none_if_the_key_does_not_exists( self, ServiceGetter: MagicMock ):
        ServiceGetter.side_effect = self.fakeLocator

        MyCache = NumpyArrayFileCache.Factory.getInstance()
        self.assertEqual(
            None,
            MyCache.get( "23" )
        )

    @patch( 'biomed.preprocessor.cache.numpyArrayFileCache.Services.getService' )
    def test_it_stores_given_data( self, ServiceGetter: MagicMock ):
        ToStore = [ 1, 2, 3 ]
        ServiceGetter.side_effect = self.fakeLocator

        MyCache = NumpyArrayFileCache.Factory.getInstance()

        MyCache.set( "1", ToStore )
        self.assertListEqual(
            ToStore,
            list( numpy.load( OS.path.join( self.__Path, "1.npy"  ) ) )
        )

        self.__Files.append( OS.path.join( self.__Path, "1.npy"  ) )

    @patch( 'biomed.preprocessor.cache.numpyArrayFileCache.Services.getService' )
    def test_it_overwrites_stored_data( self, ServiceGetter: MagicMock ):
        ToStore = [ 1, 2, 3 ]
        self.__createTestFile( ToStore, "1.npy" )
        ServiceGetter.side_effect = self.fakeLocator

        ToStore = [ 4, 5, 6 ]
        MyCache = NumpyArrayFileCache.Factory.getInstance()
        MyCache.set( "1", ToStore )
        self.assertListEqual(
            ToStore,
            list( numpy.load( OS.path.join( self.__Path, "1.npy"  ) ) )
        )

    def __del__( self ):
        self.__remove()
