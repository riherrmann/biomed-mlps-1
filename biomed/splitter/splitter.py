from abc import ABC, abstractmethod
from pandas import Series
from biomed.services_getter import ServiceGetter

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class Splitter(ABC):
    @abstractmethod
    def trainingSplit( self, X: Series, Y: Series ) -> list:
        pass

    @abstractmethod
    def validationSplit( self, X: Series, Y: Series ) -> tuple:
        pass

class SplitterFactory( ABC ):
    @abstractstatic
    def getInstance( getService: ServiceGetter ) -> Splitter:
        pass
