from abc import ABC, abstractmethod

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class FacilityManager(ABC):
    @abstractmethod
    def clean( Pmids: list, Text: list ) -> tuple:
        pass

class FacilityManagerFactory( ABC ):
    @abstractstatic
    def getInstance() -> FacilityManager:
        pass
