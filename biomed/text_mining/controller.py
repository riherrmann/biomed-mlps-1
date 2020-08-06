from abc import ABC, abstractmethod
from pandas import DataFrame

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class Controller(ABC):
    @abstractmethod
    def process(
        self,
        Data: DataFrame,
        TestData: DataFrame,
        ShortName: str,
        Description: str
    ):
        pass

class ControllerFactory( ABC ):
    @abstractstatic
    def getInstance() -> Controller:
        pass
