from abc import ABC, abstractmethod
from pandas import DataFrame
from biomed.properties_manager import PropertiesManager

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class PreProcessor(ABC):
    @abstractmethod
    def preprocess_text_corpus(self, frame: DataFrame, flags: str ) -> list:
        pass

class PreProcessorFactory( ABC ):
    @abstractstatic
    def getInstance( Properties: PropertiesManager ) -> PreProcessor:
        pass
