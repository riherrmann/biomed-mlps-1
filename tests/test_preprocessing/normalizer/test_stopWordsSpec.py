import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from biomed.preprocessor.normalizer.stopWordsFilter import StopWordsFilter
from biomed.preprocessor.normalizer.filter import Filter

class StopWordsFilterSpec( unittest.TestCase ):

    def test_it_is_a_filter( self ):
        MyFilter = StopWordsFilter.Factory.getInstance()
        self.assertTrue( isinstance( MyFilter, Filter ) )

    def test_it_ignores_non_stop_words( self ):
        MyFilter = StopWordsFilter.Factory.getInstance()
        self.assertEqual(
            "poney",
            MyFilter.apply( "poney" )
        )

    def test_it_returns_a_empty_string_if_the_given_token_is_a_stopword( self ):
        MyFilter = StopWordsFilter.Factory.getInstance()
        self.assertEqual(
            "",
            MyFilter.apply( "the" )
        )

    def test_it_removes_stopwords_independent_of_their_case( self ):
        MyFilter = StopWordsFilter.Factory.getInstance()
        self.assertEqual(
            "",
            MyFilter.apply( "ThE" )
        )
