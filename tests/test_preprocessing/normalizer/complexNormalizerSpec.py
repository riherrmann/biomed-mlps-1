import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..', 'biomed', 'preprocessor' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from normalizer.complexNormalizer import ComplexNormalizer
from normalizer.normalizer import Normalizer

class ComplexNormalizerSpec( unittest.TestCase ):

    def it_is_a_normalizer( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertTrue( isinstance( MyNormal, Normalizer ) )

    def it_filters_nouns_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            "poney text Bulloc",
            MyNormal.apply( "My little poney is writing a text for me, Bulloc.", "n" )
        )

    def it_filters_verbs_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            "write",
            MyNormal.apply( "My little poney is writing a text for me, Bulloc.", "v" )
        )

    def it_filters_adjectives_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            "little",
            MyNormal.apply( "My little poney is writing a text for me, Bulloc.", "a" )
        )

    def it_filters_mixed_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            "little poney write text Bulloc",
            MyNormal.apply( "My little poney is writing a text for me, Bulloc.", "nv" )
        )
