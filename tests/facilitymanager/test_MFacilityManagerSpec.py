import unittest
from biomed.facilitymanager.facility_manager import FacilityManager
from biomed.facilitymanager.mFacilityManager import MariosFacilityManager
from pandas import DataFrame

class MariosFacilityManagerSpec( unittest.TestCase ):

    def test_it_is_a_FacilityManager( self ):
        MyFM = MariosFacilityManager.Factory.getInstance()
        self.assertTrue( isinstance( MyFM, FacilityManager ) )

    def test_it_removes_duplicated_rows_of_a_frame( self ):
        Data = {
            "pmid": [ 1, 1, 2, 3, 1 ],
            "text": [ "aa", "aa", "bb", "cc", "aa" ],
            "is_cancer":  [ 0, 0, 1, 1, 0 ],
            "doid": [ -1, -1, 1, 2, -1 ],
            "cancer_type": [ "m", "m", "n", "n", "m" ]
        }

        Clean = {
            "pmid": [ 1, 2, 3 ],
            "text": [ "aa", "bb", "cc" ],
            "is_cancer":  [ 0, 1, 1 ],
            "doid": [ -1, 1, 2 ],
            "cancer_type": [ "m", "n", "n" ]
        }

        Frame = DataFrame(
            Data,
            columns = [ "pmid", "cancer_type", "is_cancer", "doid", "text" ]
        )

        Expected = DataFrame(
            Clean,
            columns = [ "pmid", "cancer_type", "is_cancer", "doid", "text" ]
        )

        MyFM = MariosFacilityManager.Factory.getInstance()
        Cleaned = MyFM.clean( Frame )
        print( Cleaned )
        self.assertTrue( Cleaned.equals( Expected ) )
