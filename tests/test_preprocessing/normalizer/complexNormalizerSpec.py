import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from biomed.preprocessor.normalizer.complexNormalizer import ComplexNormalizer
from biomed.preprocessor.normalizer.normalizer import Normalizer

class ComplexNormalizerSpec( unittest.TestCase ):

    def it_is_a_normalizer( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertTrue( isinstance( MyNormal, Normalizer ) )

    def it_filters_nouns_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            "poney text Bulloc",
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc." ], "n" )[ 0 ]
        )

    def it_filters_verbs_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            "write",
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc." ], "v" )[ 0 ]
        )

    def it_filters_adjectives_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            "little",
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc." ], "a" )[ 0 ]
        )

    def it_filters_mixed_out( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertEqual(
            "poney write text Bulloc",
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc." ], "nv" )[ 0 ]
        )

    def it_reads_muliple_sentences( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertListEqual(
            [ "write love" ],
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc. It loves writing." ], "v" )
        )

    def it_filters_odd_document_formats( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertListEqual(
            [ "write love" ],
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc.\n\n\n\n\nIt loves writing." ], "v" )
        )

    def it_filters_empty_sentences( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertListEqual(
            [ "little" ],
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc.\n\n\n\n\nIt loves writing." ], "a" )
        )

    def it_takes_a_stack_of_documents( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertListEqual(
            [ "write", "love", "hate", "love" ],
            MyNormal.apply( [ "My little poney is writing a text for me, Bulloc.", "It loves writing.", "But it hates cake.", "Love it!" ], "v" )
        )
