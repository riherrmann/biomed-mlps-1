from abc import ABC, abstractmethod
from pandas import DataFrame

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class FacilityManager(ABC):
    @abstractmethod
    def clean( Frame: DataFrame ) -> DataFrame:
        pass

class FacilityManagerFactory( ABC ):
    @abstractstatic
    def getInstance() -> FacilityManager:
        pass
