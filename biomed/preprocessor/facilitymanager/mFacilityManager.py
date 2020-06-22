from biomed.preprocessor.facilitymanager.facility_manager import FacilityManager
from biomed.preprocessor.facilitymanager.facility_manager import FacilityManagerFactory

class MariosFacilityManager( FacilityManager ):
    def clean( self, PmIds: list, Texts: list ) -> tuple:
        CleanedIds = list()
        CleanedTexts = list()

        while PmIds:
            Ignore = False
            PmId = PmIds.pop( 0 )
            Text = Texts.pop( 0 )

            Ignore |= self.__emptyText( Text )
            Ignore |= self.__hasDuplette( CleanedIds, PmId )

            if not Ignore:
                CleanedIds.append( PmId )
                CleanedTexts.append( Text )

        return ( CleanedIds, CleanedTexts )

    def __emptyText( self, Text: str ) -> bool:
        return False if Text.strip() else True

    def __hasDuplette( self, Cleaned: list,  Id: int ) -> bool:
        return Id in Cleaned

    class Factory( FacilityManagerFactory  ):
        @staticmethod
        def getInstance():
            return MariosFacilityManager()
