from abc import ABC, abstractmethod
from pandas import Series
from biomed.services_getter import ServiceGetter

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class Preprocessor(ABC):
    @abstractmethod
    def preprocessCorpus( self, Ids: Series, Corpus: Series ) -> Series:
        pass

class PreprocessorFactory( ABC ):
    @abstractstatic
    def getInstance( getService: ServiceGetter ) -> Preprocessor:
        pass
