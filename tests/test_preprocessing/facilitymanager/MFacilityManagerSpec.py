import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from biomed.preprocessor.facilitymanager.facility_manager import FacilityManager
from biomed.preprocessor.facilitymanager.mFacilityManager import MariosFacilityManager

class MariosFacilityManagerSpec( unittest.TestCase ):

    def it_is_a_FacilityManager( self ):
        MyFM = MariosFacilityManager.Factory.getInstance()
        self.assertTrue( isinstance( MyFM, FacilityManager ) )

    def it_removes_dupletts_by_id( self ):
        Ids = [ 12, 12, 13 ]
        Texts = [ "a", "v", "b" ]

        MyFM = MariosFacilityManager.Factory.getInstance()
        Cleaned = MyFM.clean( Ids, Texts )

        self.assertListEqual(
            Cleaned[ 0 ],
            [ 12, 13 ]
        )

        self.assertLessEqual(
            Cleaned[ 1 ],
            [ "a", "b" ]
        )

    def it_removes_empty_tuples( self ):
        Ids = [ 11, 12, 13 ]
        Texts = [ "a", "", "b" ]

        MyFM = MariosFacilityManager.Factory.getInstance()
        Cleaned = MyFM.clean( Ids, Texts )

        self.assertListEqual(
            Cleaned[ 0 ],
            [ 11, 13 ]
        )

        self.assertLessEqual(
            Cleaned[ 1 ],
            [ "a", "b" ]
        )
