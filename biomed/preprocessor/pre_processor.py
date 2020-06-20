from abc import ABC, abstractmethod
from pandas import DataFrame

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class PreProcessor(ABC):
    @abstractmethod
    def preprocess_text_corpus(self, frame: str, flags: str ) -> str:
        pass

class PreProcessorFactory( ABC ):
    @abstractstatic
    def getInstance() -> PreProcessor:
        pass
