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

    def it_ignores_unknown_flags( self ):
        Text = "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners."

        MyProc = PolymorphPreprocessor.Factory.getInstance()

        self.assertEqual(
            Text,
            MyProc.preprocess_text_corpus( Text, "opc" )
        )

    def it_uses_simple_normalizer( self ):
        Text = "Liquid chromatography with tandem mass spectrometry method for the simultaneous determination of multiple sweet mogrosides in the fruits of Siraitia grosvenorii and its marketed sweeteners. A high-performance liquid chromatography with electrospray ionization tandem mass spectrometry method has been developed and validated for the simultaneous quantification of eight major sweet mogrosides in different batches of the fruits of Siraitia grosvenorii and its marketed sweeteners."

        MyProc = PolymorphPreprocessor.Factory.getInstance()
        self.assertEqual(
            Text.lower()[ 0:100 ],
            MyProc.preprocess_text_corpus( Text, "l" )[ 0:100 ]
        )

    def it_uses_complex_normalizer( self ):
        Text = "My little cute poney is a poney"
        MyProc = PolymorphPreprocessor.Factory.getInstance()
        self.assertEqual(
            "poney poney",
            MyProc.preprocess_text_corpus( Text, "n" )
        )

    def it_uses_both_normalizers( self ):
        Text = "My little cute poney is a Poney"
        MyProc = PolymorphPreprocessor.Factory.getInstance()
        self.assertEqual(
            "poney poney",
            MyProc.preprocess_text_corpus( Text, "nl" )
        )
