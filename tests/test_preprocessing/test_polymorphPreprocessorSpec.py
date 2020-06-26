import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from unittest.mock import MagicMock
from biomed.preprocessor.normalizer.normalizer import Normalizer
from biomed.preprocessor.normalizer.normalizer import NormalizerFactory
from biomed.preprocessor.cache.cache import Cache
from biomed.preprocessor.polymorph_preprocessor import PolymorphPreprocessor
from biomed.preprocessor.pre_processor import PreProcessor
from biomed.preprocessor.facilitymanager.facility_manager import FacilityManager
from biomed.properties_manager import PropertiesManager
import numpy
from pandas import DataFrame
from multiprocessing import Manager, Lock

class StubbedFacilityManager( FacilityManager ):
    def __init__( self ):
        self.WasCalled = False
        self.ReturnEmptySet = False

    def clean( self, PmIds: list, Texts: list ) -> tuple:
        self.WasCalled = True
        if self.ReturnEmptySet:
            return ( [], [] )
        else:
            return ( PmIds, Texts )

class StubbedNormalizer( Normalizer ):
    def __init__( self ):
        self.WasCalled = False
        self.CallCounter = 0

    def apply( self, Text: str, Flags: str ) -> list:
        self.WasCalled = True
        self.CallCounter += 1
        return Text

class StubbedNormalizerFactory( NormalizerFactory ):
    def __init__( self ):
        self.CallCounter = 0
        self.LastNormalizers = list()

    def getInstance( self ):
        self.CallCounter += 1
        self.LastNormalizers.append( StubbedNormalizer() )
        return self.LastNormalizers[ len( self.LastNormalizers ) - 1 ]

class StubbedCache( Cache ):
    def __init__( self, GivenCache: dict ):
        self.__GivenCache = GivenCache
        self.WasLookedUp = False

    def has( self, Key: str ) -> bool:
        self.WasLookedUp = True
        return Key in self.__GivenCache

    def get( self, Key: str ) -> str:
        self.WasLookedUp = True
        return self.__GivenCache[ Key ]

    def set( self, Key: str, Value: str ):
        self.__GivenCache[ Key ] = Value

    def size( self ) -> int:
        return 1

    def toDict( self ) -> dict:
        return dict( self.__GivenCache )

class StubbedLock:
    def acquire():
        pass
    def release():
        pass

class PolymorphPreprocessorSpec( unittest.TestCase ):

    def __initPreprocessorDependencies( self ):
        self.__FM = StubbedFacilityManager()
        self.__FakeCache = {}
        self.__FakeCache2 = {}
        self.__Complex = StubbedNormalizerFactory()
        self.__Simple = StubbedNormalizerFactory()
        self.__SimpleFlags = [ "s", "l", "w" ]
        self.__ComplexFlags = [ "n", "v", "a" ]
        self.__Shared = StubbedCache( self.__FakeCache )
        self.__FileCache = StubbedCache( self.__FakeCache2 )

    def setUp( self ):
        self.__initPreprocessorDependencies()

        self.__Prepro = PolymorphPreprocessor(
            self.__FM,
            1,
            self.__FileCache,
            self.__Shared,
            self.__Simple,
            self.__SimpleFlags,
            self.__Complex,
            self.__ComplexFlags,
            MagicMock( spec=StubbedLock )
       )

    def test_it_is_a_PreProcessor( self ):
        Path = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), 'testTmp' ) )
        OS.mkdir( Path, 0o777 )
        PM = PropertiesManager()
        PM.cache_dir = Path
        MyProc = PolymorphPreprocessor.Factory.getInstance( PM )
        self.assertTrue( isinstance( MyProc, PreProcessor ) )

    def test_it_does_not_alter_the_source( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }
        Source = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "" )

        self.assertDictEqual(
            TestData,
            Source
        )

    def test_it_ignores_unknown_flags( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "opc" )

        self.assertFalse( self.__Simple.LastNormalizers[ 0 ].WasCalled )
        self.assertFalse( self.__Complex.LastNormalizers[ 0 ].WasCalled )

    def test_it_uses_simple_normalizer( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "l" )

        self.assertTrue( self.__Simple.LastNormalizers[ 0 ].WasCalled )
        self.assertFalse( self.__Complex.LastNormalizers[ 0 ].WasCalled )

    def test_it_uses_complex_normalizer( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "My little cute poney is a poney" ]
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "n" )

        self.assertFalse( self.__Simple.LastNormalizers[ 0 ].WasCalled )
        self.assertTrue( self.__Complex.LastNormalizers[ 0 ].WasCalled )

    def test_it_uses_both_normalizers( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "My little cute Poney is a Poney" ]
        }
        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "nl" )

        self.assertTrue( self.__Simple.LastNormalizers[ 0 ].WasCalled )
        self.assertTrue( self.__Complex.LastNormalizers[ 0 ].WasCalled )

    def test_it_iterates_over_all_given_texts( self ):
        TestData = {
            'pmid': [ 42, 41, 40 ],
            'cancer_type': [ -1, -1, -1 ],
            'doid': [ 23, 22, 21 ],
            'is_cancer': [ False, False, False ],
            'text': [
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs."
            ]
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "nl" )

        self.assertEqual(
            1,
            self.__Complex.LastNormalizers[ 0 ].CallCounter
        )

        self.assertEqual(
            1,
            self.__Simple.LastNormalizers[ 0 ].CallCounter
        )


    def test_it_uses_a_cache_to_determine_if_the_value_was_already_processed( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        self.__FakeCache[ "42a" ] =  TestData[ "text"][ 0 ].lower()

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        Result = self.__Prepro.preprocess_text_corpus( MyFrame, "a" )
        self.assertFalse( self.__Complex.LastNormalizers[ 0 ].WasCalled )
        self.assertFalse( self.__Simple.LastNormalizers[ 0 ].WasCalled )
        self.assertEqual(
             Result[ 0 ],
             self.__FakeCache[ "42a" ]
         )

    def test_it_caches_new_text_variants( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )

        self.__Prepro.preprocess_text_corpus( MyFrame, "a" )
        self.assertTrue( self.__Complex.LastNormalizers[ 0 ].WasCalled )
        self.assertTrue( "42a" in self.__FakeCache )
        self.assertEqual(
            TestData[ "text" ][ 0 ],
            self.__FakeCache[ "42a" ]
        )

    def test_it_does_not_run_in_parallel_if_only_one_worker_is_given( self ):
        TestData = {
            'pmid': [ 52, 51, 50, 39, 38, 37, 35, 34, 33, 32, 31, 30 ],
            'text': [
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs.",
                "My little farm is cute.",
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs.",
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs.",
                "My little farm is cute.",
                "My little cute Poney is a Poney",
            ]
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__initPreprocessorDependencies()
        self.__Prepro = PolymorphPreprocessor(
            self.__FM,
            1,
            self.__FileCache,
            self.__Shared,
            self.__Simple,
            self.__SimpleFlags,
            self.__Complex,
            self.__ComplexFlags,
            MagicMock( spec=StubbedLock )
        )

        self.__Prepro.preprocess_text_corpus( MyFrame, "al" )

        self.assertEqual(
            1,
            self.__Simple.CallCounter
        )

        self.assertEqual(
            1,
            self.__Complex.LastNormalizers[ 0 ].CallCounter
        )

        self.assertEqual(
            1,
            self.__Simple.LastNormalizers[ 0 ].CallCounter
        )

    def test_it_runs_in_parallel( self ):
        TestData = {
            'pmid': [ 52, 51, 50, 39, 38, 37, 35, 34, 33, 32, 31, 30 ],
            'text': [
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs.",
                "My little farm is cute.",
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs.",
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs.",
                "My little farm is cute.",
                "My little cute Poney is a Poney",
            ]
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__initPreprocessorDependencies()
        self.__FakeCache = Manager().dict()
        self.__Shared = StubbedCache( self.__FakeCache )

        self.__Prepro = PolymorphPreprocessor(
            self.__FM,
            3,
            self.__FileCache,
            self.__Shared,
            self.__Simple,
            self.__SimpleFlags,
            self.__Complex,
            self.__ComplexFlags,
            MagicMock( spec=StubbedLock )
        )

        self.__Prepro.preprocess_text_corpus( MyFrame, "al" )
        self.assertEqual(
            3,
            len( self.__Complex.LastNormalizers )
        )
        self.assertEqual(
            3,
            len( self.__Simple.LastNormalizers )
        )

        for Normierer in self.__Complex.LastNormalizers:
            self.assertFalse( Normierer.WasCalled )
        for Normierer in self.__Simple.LastNormalizers:
            self.assertFalse( Normierer.WasCalled )

        Parsed = self.__FakeCache.values()
        for Text in MyFrame[ "text" ]:
            self.assertTrue( Text in Parsed )

    def test_it_clean_up_the_data( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "l" )

        self.assertTrue( self.__FM.WasCalled )

    def test_it_fails_on_empty_dataset( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__FM.ReturnEmptySet = True
        with self.assertRaises( RuntimeError ):
            self.__Prepro.preprocess_text_corpus( MyFrame, "l" )

    def test_it_saves_the_shared_memory_on_a_cache_miss_after_the_computing_stage( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "I love my poney." ],
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "l" )

        self.assertDictEqual(
            self.__FakeCache,
            self.__FakeCache2[ "hardId42" ]
        )

    def test_it_loads_shared_memory_on_init( self ):
        Path = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), 'testTmp' ) )
        File = OS.path.join( Path, "hardId42.npy" )
        Saved = { "42l": "stop words" }
        OS.mkdir( Path, 0o777 )
        numpy.save( File, Saved )
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        PM = PropertiesManager()
        PM.cache_dir = Path
        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        MyProc = PolymorphPreprocessor.Factory.getInstance( PM )
        Results = MyProc.preprocess_text_corpus( MyFrame, "l" )
        del MyProc

        self.assertEqual(
            Saved[ "42l" ],
            Results[ 0 ]
        )

    def tearDown( self ):
        Path = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), 'testTmp' ) )
        File = OS.path.join( Path, "hardId42.npy" )
        if OS.path.isfile( File ):
            OS.remove( File )

        if OS.path.isdir( Path ):
            OS.rmdir( Path )
