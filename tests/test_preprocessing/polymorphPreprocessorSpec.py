import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', 'biomed', 'preprocessor' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from polymorph_preprocessor import PolymorphPreprocessor
from pre_processor import PreProcessor
from pandas import DataFrame

class PolymorphPreprocessorSpec( unittest.TestCase ):
    def it_is_a_PreProcessor( self ):
        MyProc = PolymorphPreprocessor.Factory.getInstance()
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
        MyProc = PolymorphPreprocessor.Factory.getInstance()
        MyProc.preprocess_text_corpus( MyFrame, "" )

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
        MyProc = PolymorphPreprocessor.Factory.getInstance()
        Result = MyProc.preprocess_text_corpus( MyFrame, "opc" )

        self.assertEqual(
            Result[ 0 ],
            TestData[ "text" ][ 0 ]
        )

    def it_uses_simple_normalizer( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners." ],
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        MyProc = PolymorphPreprocessor.Factory.getInstance()
        Result = MyProc.preprocess_text_corpus( MyFrame, "l" )

        self.assertEqual(
            Result[ 0 ][ 0:100 ],
            TestData[ "text" ][ 0 ].lower()[ 0: 100 ]
        )

    def it_uses_complex_normalizer( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "My little cute poney is a poney" ]
        }

        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        MyProc = PolymorphPreprocessor.Factory.getInstance()
        Result = MyProc.preprocess_text_corpus( MyFrame, "n" )
        self.assertEqual(
            "poney poney",
            Result[ 0 ]
        )

    def it_uses_both_normalizers( self ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "My little cute Poney is a Poney" ]
        }
        MyProc = PolymorphPreprocessor.Factory.getInstance()
        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        Result = MyProc.preprocess_text_corpus( MyFrame, "nl" )
        self.assertEqual(
            "poney poney",
            Result[ 0 ]
        )

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
        Expected = [ "poney poney", "farm", "programm application" ]

        MyProc = PolymorphPreprocessor.Factory.getInstance()
        MyFrame = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        Result = MyProc.preprocess_text_corpus( MyFrame, "nl" )

        self.assertListEqual(
            Result,
            Expected
        )

    def it_uses_a_cache_to_determine_if_the_value_was_already_processed( self ):
        pass


Suite = unittest.TestSuite()
Suite.addTests( [
    PolymorphPreprocessorSpec( "it_is_a_PreProcessor" ),
    PolymorphPreprocessorSpec( "it_does_not_alter_the_source" ),
    PolymorphPreprocessorSpec( "it_ignores_unknown_flags" ),
    PolymorphPreprocessorSpec( "it_uses_simple_normalizer" ),
    PolymorphPreprocessorSpec( "it_uses_complex_normalizer" ),
    PolymorphPreprocessorSpec( "it_uses_both_normalizers" ),
    PolymorphPreprocessorSpec( "it_iterates_over_all_given_texts" ),
] )


Runner = unittest.TextTestRunner()
Runner.run( Suite )
