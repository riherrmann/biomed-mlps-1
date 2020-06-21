import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from biomed.preprocessor.normalizer.stemFilter import StemFilter
from biomed.preprocessor.normalizer.filter import Filter

class StemFilterSpec( unittest.TestCase ):

    def it_is_a_filter( self ):
        MyFilter = StemFilter.Factory.getInstance()
        self.assertTrue( isinstance( MyFilter, Filter ) )

    def it_stems_a_given_word( self ):
        MyFilter = StemFilter.Factory.getInstance()
        self.assertEqual(
            "write",
            MyFilter.apply( "writing" )
        )
