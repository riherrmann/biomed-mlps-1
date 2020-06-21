import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from biomed.preprocessor.normalizer.normalizer import Normalizer
from biomed.preprocessor.cache.cache import Cache
from biomed.preprocessor.polymorph_preprocessor import PolymorphPreprocessor
from biomed.preprocessor.pre_processor import PreProcessor
from biomed.properties_manager import PropertiesManager
from pandas import DataFrame

class StubbedNormalizer( Normalizer ):
    def __init__( self ):
        self.WasCalled = False
        self.CallCounter = 0

    def apply( self, Text: str, Flags: str ) -> list:
        self.WasCalled = True
        self.CallCounter += 1
        return Text

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

class PolymorphPreprocessorSpec( unittest.TestCase ):
    def setUp( self ):
        self.__FakeCache = {}
        self.__FakeCache2 = {}
        self.__Complex = StubbedNormalizer()
        self.__Simple = StubbedNormalizer()
        self.__SimpleFlags = [ "s", "l", "w" ]
        self.__ComplexFlags = [ "n", "v", "a" ]
        self.__Shared = StubbedCache( self.__FakeCache )
        self.__FileCache = StubbedCache( self.__FakeCache2 )

        self.__Prepro = PolymorphPreprocessor(
            1,
            self.__FileCache,
            self.__Shared,
            self.__Simple,
            self.__SimpleFlags,
            self.__Complex,
            self.__ComplexFlags
       )

    def it_is_a_PreProcessor( self ):
        MyProc = PolymorphPreprocessor.Factory.getInstance( PropertiesManager() )
        self.assertTrue( isinstance( MyProc, PreProcessor ) )

    def it_does_not_alter_the_source( self ):
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

    def it_ignores_unknown_flags( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "opc" )

        self.assertFalse( self.__Simple.WasCalled )
        self.assertFalse( self.__Complex.WasCalled )

    def it_uses_simple_normalizer( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "l" )

        self.assertTrue( self.__Simple.WasCalled )
        self.assertFalse( self.__Complex.WasCalled )

    def it_uses_complex_normalizer( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "My little cute poney is a poney" ]
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "n" )

        self.assertFalse( self.__Simple.WasCalled )
        self.assertTrue( self.__Complex.WasCalled )

    def it_uses_both_normalizers( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "My little cute Poney is a Poney" ]
        }
        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "nl" )

        self.assertTrue( self.__Simple.WasCalled )
        self.assertTrue( self.__Complex.WasCalled )

    def it_iterates_over_all_given_texts( self ):
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
            len( TestData[ "text" ] ),
            self.__Complex.CallCounter
        )

        self.assertEqual(
            len( TestData[ "text" ] ),
            self.__Simple.CallCounter
        )


    def it_uses_a_cache_to_determine_if_the_value_was_already_processed( self ):
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
        self.assertFalse( self.__Complex.WasCalled )
        self.assertFalse( self.__Simple.WasCalled )
        self.assertEqual(
             Result[ 0 ],
             self.__FakeCache[ "42a" ]
         )

    def it_caches_new_text_variants( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )

        self.__Prepro.preprocess_text_corpus( MyFrame, "a" )
        self.assertTrue( self.__Complex.WasCalled )
        self.assertTrue( "42a" in self.__FakeCache )
        self.assertEqual(
            TestData[ "text" ][ 0 ],
            self.__FakeCache[ "42a" ]
        )

    def it_does_not_lookup_the_cache_if_no_variant_is_applicable( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "opc" )

        self.assertFalse( self.__FileCache.WasLookedUp )

    def it_looks_up_on_known_applicable_variants( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "l" )

        self.assertTrue( self.__FileCache.WasLookedUp )

    def it_returns_the_value_of_the_file_cache( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        self.__FakeCache2[ "0e2f1a75f3af555d48a593a9e0a610ee" ] =  [ TestData[ "text"][ 0 ].lower() ]

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        Result = self.__Prepro.preprocess_text_corpus( MyFrame, "a" )

        self.assertFalse( self.__Complex.WasCalled )
        self.assertFalse( self.__Simple.WasCalled )
        self.assertListEqual(
             Result,
             self.__FakeCache2[ "0e2f1a75f3af555d48a593a9e0a610ee" ]
         )

    def it_caches_new_set_variants( self ):
        TestData = {
            'pmid': [ 42, 41, 40],
            'cancer_type': [ -1, -1, -1 ],
            'doid': [ 23, 23, 21 ],
            'is_cancer': [ False, False, False ],
            'text': [
                "My little cute Poney is a Poney",
                "My little farm is cute.",
                "My little programm is a application and runs and runs and runs."
            ]
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        self.__Prepro.preprocess_text_corpus( MyFrame, "al" )

        self.assertTrue( self.__Complex.WasCalled )
        self.assertTrue( self.__Simple.WasCalled )
        self.assertListEqual(
            TestData[ "text" ],
            self.__FakeCache2[ "a88cc70d078ce3d60e7b51757cda82c7" ]
        )

    def it_runs_in_paralell( self ):
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

        self.__Prepro = PolymorphPreprocessor(
            3,
            self.__FileCache,
            self.__Shared,
            self.__Simple,
            self.__SimpleFlags,
            self.__Complex,
            self.__ComplexFlags
        )

        self.__Prepro.preprocess_text_corpus( MyFrame, "al" )

        self.assertTrue( self.__Complex.WasCalled )
        self.assertTrue( self.__Simple.WasCalled )
        # TODO
