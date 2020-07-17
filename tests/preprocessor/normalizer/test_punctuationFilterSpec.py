import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from biomed.preprocessor.normalizer.punctuationFilter import PunctuationFilter
from biomed.preprocessor.normalizer.filter import Filter

class PunctuationFilterSpec( unittest.TestCase ):

    def test_it_is_a_filter( self ):
        MyFilter = PunctuationFilter.Factory.getInstance()
        self.assertTrue( isinstance( MyFilter, Filter ) )

    def test_it_removes_punctuations( self ):
        MyFilter = PunctuationFilter.Factory.getInstance()
        self.assertEqual(
            '',
            MyFilter.apply( '.?"$%&~!' )
        )
