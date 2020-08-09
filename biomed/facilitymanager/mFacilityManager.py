from biomed.facilitymanager.facility_manager import FacilityManager
from biomed.facilitymanager.facility_manager import FacilityManagerFactory
from pandas import DataFrame

class MariosFacilityManager( FacilityManager ):
    def clean( self, Frame: DataFrame ) -> DataFrame:
        Columns = list( Frame.head() )
        Ids = []
        CleanRows = []

        for Index in range( 0, len( Frame.index ) ):
            Row =  Frame.loc[ [ Index ], : ]
            Id = self.__getId( Index, Row, Columns )
            if not self.__hasDuplette( Ids, Id ):
                CleanRows.append( self.__makeRow( Index, Row, Columns ) )
                Ids.append( Id )

        return DataFrame( CleanRows, columns = Columns )

    def __makeRow( self, Index: int, Row: DataFrame, Columns: list ) -> list:
        NewRow = []
        for Column in Columns:
            NewRow.append( list( Row[ Column ] )[ 0 ] )

        return NewRow

    def __getId( self, Index: int, Row: DataFrame, Columns: list ):
         return list( Row[ Columns[ 0 ] ] )[ 0 ]

    def __hasDuplette( self, Cleaned: list,  Id: int ) -> bool:
         return Id in Cleaned

    class Factory( FacilityManagerFactory  ):
        @staticmethod
        def getInstance():
            return MariosFacilityManager()
