import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..', 'biomed', 'preprocessor' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from normalizer.lowerFilter import LowerFilter
from normalizer.filter import Filter

class LowerFilterSpec( unittest.TestCase ):

    def it_is_a_filter( self ):
        MyFilter = LowerFilter.Factory.getInstance()
        self.assertTrue( isinstance( MyFilter, Filter ) )

    def it_brings_a_given_token_into_lower_case( self ):
        MyFilter = LowerFilter.Factory.getInstance()
        self.assertEqual(
            "uppercase",
            MyFilter.apply( "UpperCase" )
        )
