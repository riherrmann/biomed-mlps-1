import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..', 'biomed', 'preprocessor' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from punctuationFilter import PunctuationFilter
from filter import Filter

class PunctuationFilterSpec( unittest.TestCase ):

    def it_is_a_filter( self ):
        MyFilter = PunctuationFilter.Factory.getInstance()
        self.assertTrue( isinstance( MyFilter, Filter ) )

    def it_removes_punctuations( self ):
        MyFilter = PunctuationFilter.Factory.getInstance()
        self.assertEqual(
            '',
            MyFilter.apply( '.?"$%&~!' )
        )
