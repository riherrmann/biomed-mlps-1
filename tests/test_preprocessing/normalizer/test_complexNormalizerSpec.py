import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from biomed.preprocessor.normalizer.complexNormalizer import ComplexNormalizer
from biomed.preprocessor.normalizer.normalizer import Normalizer

class ComplexNormalizerSpec( unittest.TestCase ):

    def test_it_is_a_normalizer( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertTrue( isinstance( MyNormal, Normalizer ) )

    def test_it_filters_nouns_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            "poney text Bulloc",
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc." ], "n" )[ 0 ]
        )

    def test_it_filters_verbs_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            "write",
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc." ], "v" )[ 0 ]
        )

    def test_it_filters_adjectives_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            "little",
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc." ], "a" )[ 0 ]
        )

    def test_it_filters_symbols_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            "$",
            MyNormal.apply( [ "And then he just found $1." ], "y" )[ 0 ]
        )

    def test_it_filters_numerals_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            "3.14159265359",
            MyNormal.apply( [ "And then he just found 3.14159265359." ], "u" )[ 0 ]
        )


    def test_it_filters_mixed_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            "poney write text Bulloc",
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc." ], "nv" )[ 0 ]
        )

    def test_it_reads_muliple_sentences( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertListEqual(
            [ "write love" ],
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc. It loves writing." ], "v" )
        )

    def test_it_filters_odd_document_formats( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertListEqual(
            [ "write love" ],
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc.\n\n\n\n\nIt loves writing." ], "v" )
        )

    def test_it_filters_empty_sentences( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertListEqual(
            [ "little" ],
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc.\n\n\n\n\nIt loves writing." ], "a" )
        )

    def test_it_takes_a_stack_of_documents( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertListEqual(
            [ "write", "love", "hate", "love" ],
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc.", "It loves writing.", "But it hates cake.", "Love it!" ], "v" )
        )
