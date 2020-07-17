import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from biomed.preprocessor.normalizer.simpleNormalizer import SimpleNormalizer
from biomed.preprocessor.normalizer.normalizer import Normalizer

class SimpleNormalizerSpec( unittest.TestCase ):

    def test_it_is_a_normalizer( self ):
        MyNormal = SimpleNormalizer.Factory.getInstance()
        self.assertTrue( isinstance( MyNormal, Normalizer ) )

    def test_it_brings_all_chars_to_lower_case( self ):
        MyNormal = SimpleNormalizer.Factory.getInstance()
        self.assertEqual(
            "my little poney",
            MyNormal.apply( [ "My Little Poney." ], "l" )[ 0 ]
        )

    def test_it_removes_stopwords( self ):
        MyNormal = SimpleNormalizer.Factory.getInstance()
        self.assertEqual(
            "text stop words",
            MyNormal.apply( [ "A text of stop words." ], "w" )[ 0 ]
        )

    def test_it_stems_words( self ):
        MyNormal = SimpleNormalizer.Factory.getInstance()
        self.assertEqual(
            "My poney write text",
            MyNormal.apply( [ "My poney writes texts." ], "s" )[ 0 ]
        )

    def test_it_uses_mutiple_filter( self ):
        MyNormal = SimpleNormalizer.Factory.getInstance()
        self.assertEqual(
            "poney write text stop word",
            MyNormal.apply( [ "My Poney writes Texts of Stop words." ], "slw" )[ 0 ]
        )

    def test_it_handles_multiple_senetences( self ):
        MyNormal = SimpleNormalizer.Factory.getInstance()
        self.assertEqual(
            "My poney write a text It is happi about that and it is write more" ,
            MyNormal.apply( [ "My poney writes a texts. It is happy about that. And it is writing more." ], "s" )[ 0 ]
        )

    def test_it_handles_a_batch_of_documents( self ):
        MyNormal = SimpleNormalizer.Factory.getInstance()
        self.assertListEqual(
            ["My poney write a text", "It is happi about it", "and it is write more" ] ,
            MyNormal.apply( [ "My poney writes a texts.", "It is happy about it.", "And it is writing more." ], "s" )
        )
