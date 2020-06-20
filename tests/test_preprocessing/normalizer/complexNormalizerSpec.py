import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..', 'biomed', 'preprocessor', 'normalizer' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from complexNormalizer import ComplexNormalizer
from normalizer import Normalizer

class ComplexNormalizerSpec( unittest.TestCase ):

    def it_is_a_normalizer( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertTrue( isinstance( MyNormal, Normalizer ) )

    def it_filters_nouns_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            [ "poney", "text", "Bulloc" ],
            MyNormal.apply( "My little poney is writing a text for me, Bulloc.", "n" )
        )

    def it_filters_verbs_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            [ "write" ],
            MyNormal.apply( "My little poney is writing a text for me, Bulloc.", "v" )
        )

    def it_filters_mixed_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            [ "poney", "write", "text", "Bulloc" ],
            MyNormal.apply( "My little poney is writing a text for me, Bulloc.", "nv" )
        )
