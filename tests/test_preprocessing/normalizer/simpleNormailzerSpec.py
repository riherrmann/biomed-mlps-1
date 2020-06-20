import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..', 'biomed', 'preprocessor' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from simpleNormalizer import SimpleNormalizer
from normalizer import Normalizer

class SimpleNormalizerSpec( unittest.TestCase ):

    def it_is_a_normalizer( self ):
        MyNormal = SimpleNormalizer.Factory.getInstance()
        self.assertTrue( isinstance( MyNormal, Normalizer ) )

    def it_brings_all_chars_to_lower_case( self ):
        MyNormal = SimpleNormalizer.Factory.getInstance()
        self.assertEquals(
            [ "my", "little", "poney" ],
            MyNormal.apply( "My Little Poney.", "l" )
        )

    def it_removes_stopwords( self ):
        MyNormal = SimpleNormalizer.Factory.getInstance()
        self.assertEquals(
            [ "text", "stop", "words" ],
            MyNormal.apply( "A text of stop words.", "w" )
        )

    def it_stems_words( self ):
        MyNormal = SimpleNormalizer.Factory.getInstance()
        self.assertEquals(
            [ "My", "poney", "write", "text" ],
            MyNormal.apply( "My poney writes texts.", "s" )
        )

    def it_uses_mutiple_filter( self ):
        MyNormal = SimpleNormalizer.Factory.getInstance()
        self.assertEquals(
            [ "poney", "write", "text", "stop", "word" ],
            MyNormal.apply( "My Poney writes Texts of Stop words.", "slw" )
        )
